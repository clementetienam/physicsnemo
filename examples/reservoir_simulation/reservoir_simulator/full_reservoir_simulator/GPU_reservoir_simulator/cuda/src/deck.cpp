// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// deck.cpp — implementation of the OPM/Eclipse-style keyword deck parser.
#include "deck.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>

namespace nvrs {

int Deck::n_prod() const {
  int c = 0;
  for (const auto& w : wells)
    if (w.type < 0) ++c;
  return c;
}
int Deck::n_inj() const {
  int c = 0;
  for (const auto& w : wells)
    if (w.type > 0) ++c;
  return c;
}

namespace {

// Strip `--` comments, trim, and upper-case-detect keywords.
std::string strip_comment(const std::string& line) {
  auto pos = line.find("--");
  std::string s = (pos == std::string::npos) ? line : line.substr(0, pos);
  // trim
  auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

bool is_keyword(const std::string& s) {
  if (s.empty()) return false;
  for (char c : s)
    if (!(std::isupper((unsigned char)c) || std::isdigit((unsigned char)c) ||
          c == '_'))
      return false;
  return std::isalpha((unsigned char)s[0]);
}

std::vector<std::string> tokens(const std::string& s) {
  std::vector<std::string> t;
  std::istringstream is(s);
  std::string w;
  while (is >> w) t.push_back(w);
  return t;
}

Real toR(const std::string& s) { return std::stod(s); }
int  toI(const std::string& s) { return std::stoi(s); }

}  // namespace

Deck parse_deck(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open deck: " + path);

  Deck d;
  std::string raw;
  std::string kw;                 // current active keyword/section
  std::vector<std::string> recs;  // accumulated records for list keywords

  auto flush_records = [&](const std::string& key,
                           std::vector<std::string>& rows) {
    if (rows.empty()) return;
    if (key == "SWOF" || key == "SWOW") {
      for (auto& r : rows) {
        auto tk = tokens(r);
        if (tk.size() >= 3) {
          d.swof.push_back({toR(tk[0]), toR(tk[1]), toR(tk[2])});
          d.swof_pcow.push_back(tk.size() >= 4 ? toR(tk[3]) : (Real)0);
        }
      }
    } else if (key == "SGOF" || key == "SWOG") {
      for (auto& r : rows) {
        auto tk = tokens(r);
        if (tk.size() >= 3) {
          d.sgof.push_back({toR(tk[0]), toR(tk[1]), toR(tk[2])});
          d.sgof_pcog.push_back(tk.size() >= 4 ? toR(tk[3]) : (Real)0);
        }
      }
    } else if (key == "WELLS" || key == "WELSPECS") {
      // I  J  TYPE  RATE  RW  SKIN  PWF  PERF  NAME
      for (auto& r : rows) {
        auto tk = tokens(r);
        if (tk.size() < 4) continue;
        WellSpec w;
        w.i = toI(tk[0]);
        w.j = toI(tk[1]);
        w.type = toI(tk[2]);
        w.rate = toR(tk[3]);
        if (tk.size() > 4) w.rw = toR(tk[4]);
        if (tk.size() > 5) w.skin = toR(tk[5]);
        if (tk.size() > 6) w.pwf = toR(tk[6]);
        if (tk.size() > 7) w.perf = tk[7];
        if (tk.size() > 8) w.name = tk[8];
        else w.name = (w.type > 0 ? "I" : "P") + std::to_string(d.wells.size());
        d.wells.push_back(w);
      }
    } else if (key == "ZMF") {          // compositional feed composition
      for (auto& r : rows)
        for (auto& t : tokens(r)) d.comp_z.push_back(toR(t));
    } else if (key == "TCRIT") {
      for (auto& r : rows) for (auto& t : tokens(r)) d.comp_Tc.push_back(toR(t));
    } else if (key == "PCRIT") {
      for (auto& r : rows) for (auto& t : tokens(r)) d.comp_Pc.push_back(toR(t));
    } else if (key == "ACF") {
      for (auto& r : rows) for (auto& t : tokens(r)) d.comp_acf.push_back(toR(t));
    } else if (key == "VMOL") {
      for (auto& r : rows) for (auto& t : tokens(r)) d.comp_vmL.push_back(toR(t));
    }
    rows.clear();
  };

  // Scalar key=value handler used inside RUNSPEC/GRID/PROPS/SOLVER/SCHEDULE.
  // Scalar keywords, part two: solver, timestep, well and schedule
  // controls.
  //
  // Split out of set_scalar rather than continuing it because an else-if
  // chain costs one level of block nesting per branch, and MSVC stops at
  // 128 (C1061).  The single chain reached ~115 branches and the next
  // keyword added would not compile.  set_scalar falls through to this on
  // any key it does not match, so the two together behave exactly as the
  // one chain did.
  auto set_scalar_ctl = [&](const std::string& k, const std::string& v) {
    auto S = [&](Real& dst) { dst = toR(v); };
    auto I = [&](int& dst) { dst = toI(v); };
    if (k == "AIM") { I(d.aim); d.aim_explicit = 1; }
    else if (k == "AIMCFL") S(d.aim_cfl);
    else if (k == "IMPLICIT") {
      if (v == "AIM" || v == "1") { d.aim = 1; d.aim_explicit = 1; }
      else if (v == "FIM" || v == "FULL" || v == "0") {
        d.aim = 0; d.aim_explicit = 1;
      }
    }
    else if (k == "STAGE2") {
      if (v == "ZLINE" || v == "1") d.stage2 = 1;
      else if (v == "JACOBI" || v == "0") d.stage2 = 0;
      else if (v == "GINKGO" || v == "ADAPTIVE" || v == "2") d.stage2 = 2;
      else
        throw std::runtime_error("STAGE2 must be JACOBI, ZLINE or GINKGO, got '" +
                                 v + "'");
    }
    else if (k == "MIXED") I(d.mixed_precision);
    else if (k == "EMULATE") I(d.emulate_fp64);
    else if (k == "BLOCKSPMV") I(d.block_spmv);
    else if (k == "CUDAGRAPH") I(d.cuda_graph);
    else if (k == "SCHEME") {
      if (v == "SEQUENTIAL" || v == "SEQ" || v == "IMPES" || v == "1") d.scheme = 1;
      else if (v == "FIM" || v == "IMPLICIT" || v == "0") d.scheme = 0;
    }
    else if (k == "JACOBIAN") {
      // ANALYTIC and NUMERICAL are the two assemblies that have always been
      // here.  TENSORCORE is the operator-product path: the local connection
      // Jacobian is formed as L_flux * L_props and the batch is run as one
      // padded GEMM, on Tensor Cores where the device has them.  Anything
      // unrecognised is an error rather than a silent fall back to the
      // default, for the same reason as LINSOLVER.
      if (v == "ANALYTIC" || v == "1") d.jacobian = 1;
      else if (v == "NUMERICAL" || v == "0") d.jacobian = 0;
      else if (v == "TENSORCORE" || v == "TC" || v == "2") d.jacobian = 2;
      else
        throw std::runtime_error(
            "JACOBIAN must be NUMERICAL, ANALYTIC or TENSORCORE, got '" + v +
            "'");
    }
    else if (k == "TCVALIDATE") I(d.tc_validate);
    else if (k == "TCPRECISION") {
      std::string u = v;
      for (auto& c : u) c = (char)std::toupper((unsigned char)c);
      if (u == "TF32") d.tc_precision = 32;
      else if (u == "FP32" || u == "SIMT") d.tc_precision = 0;
      else
        throw std::runtime_error("TCPRECISION must be TF32 or FP32, got '" + v +
                                 "'");
    }
    else if (k == "DTREPORT") S(d.dt_report);
    else if (k == "DTMAX") S(d.dt_max);
    else if (k == "DTGROW") I(d.dt_grow);
    else if (k == "DTKEEP") I(d.dt_keep);
    else if (k == "DTRELAX") I(d.dt_relax);
    else if (k == "DTTARGET") I(d.dt_target);
    else if (k == "DTCONTROL") I(d.dt_control);
    else if (k == "DTRESTART") S(d.dt_restart);
    else if (k == "DTMAXGROW") S(d.dt_max_grow);
    else if (k == "DTDECAYDAMP") S(d.dt_decay_damp);
    else if (k == "DTGROWDAMP") S(d.dt_grow_damp);
    else if (k == "DTPID") I(d.dt_pid);
    else if (k == "DTPIDTOL") S(d.dt_pid_tol);
    else if (k == "DTCAPPROBE") I(d.dt_cap_probe);
    else if (k == "DTCAPPROBEITERS") I(d.dt_cap_probe_iters);
    else if (k == "DTCAPPROBEFACTOR") S(d.dt_cap_probe_factor);
    else if (k == "DTFAILCAP") S(d.dt_fail_cap);
    else if (k == "NEWTONSTALLMAX") I(d.newton_stall_max);
    else if (k == "OPMCONV") I(d.opm_conv);
    else if (k == "CNVRELAXPV") S(d.cnv_relax_pv);
    else if (k == "CNVRELAXED") S(d.cnv_relaxed);
    else if (k == "DTCAPGROW") S(d.dt_cap_grow);
    else if (k == "TEND") S(d.t_end);
    else if (k == "SUBSTEPS") I(d.newton_substeps);
    else if (k == "NEWTONMAX") I(d.newton_max);
    else if (k == "MBTOL") S(d.mb_tol);
    else if (k == "MBTOLRELAXED") S(d.mb_tol_relaxed);
    else if (k == "CNVTOL") S(d.cnv_tol);
    else if (k == "NEWTONSTALL") S(d.newton_stall);
    else if (k == "NEWTONTOL") S(d.newton_tol);
    else if (k == "LINESEARCH") I(d.line_search);
    else if (k == "LINEBEST") I(d.line_best);
    else if (k == "LINFORCINGMAX") S(d.lin_forcing_max);
    else if (k == "DPMAX") S(d.dp_max);
    else if (k == "DPMAXREL") S(d.dp_max_rel);
    else if (k == "METRIC") d.metric_input = 1;
    else if (k == "FIELD")  d.metric_input = 0;
    else if (k == "UNITS")  d.metric_input = (v == "METRIC" || v == "metric");
    else if (k == "DSMAX") S(d.ds_max);
    else if (k == "MAXCHOP") I(d.max_chop);
    else if (k == "MAKECSV") I(d.make_csv);
    else if (k == "MAKEVTK") I(d.make_vtk);
    else if (k == "MAKESUMMARY") I(d.make_summary);
    else if (k == "MAKEGASSMANN") I(d.make_gassmann);
    else if (k == "AUTOPLOT") I(d.autoplot);
    else if (k == "ECLOUT") I(d.ecl_out);
    else if (k == "CASENAME") d.case_name = v;
    else if (k == "TITLE") d.title = v;
    else if (k == "PERMFILE") d.perm_path = v;
    else if (k == "FIELDDECK") d.field_deck = v;
    // One keyword per gradient.  The scalar parser above takes only the first
    // token after a key, so a single GRADIENTS with three numbers would read
    // one and silently discard the rest -- and a missing gradient does not
    // fail, it just skips equilibration and leaves the field at PINIT.
    else if (k == "GRADOIL")   { d.grad_oil = toR(v); d.have_gradients = true; }
    else if (k == "GRADWATER") { d.grad_water = toR(v); }
    else if (k == "GRADGAS")   { d.grad_gas = toR(v); }
    else if (k == "SWCONNATE") S(d.connate_water);
    else if (k == "USESCHEDULE") I(d.use_schedule);
    else if (k == "REPORTDATES") I(d.report_dates);
    else if (k == "PRODBHP") S(d.prod_bhp);
    else if (k == "PRODDRAW") S(d.prod_draw_max);
    else if (k == "WELLXFLOW") I(d.well_crossflow);
    else if (k == "INJBHP") S(d.inj_bhp);
    else if (k == "WELLITER") I(d.well_iter);
    else if (k == "WELLSCHUR") I(d.well_schur);
    else if (k == "PVSWITCH") I(d.pv_switch);
    else if (k == "POROFILE") d.poro_path = v;
    else if (k == "PERMZFILE") d.permz_path = v;
    else if (k == "OUTDIR") d.out_dir = v;
    else if (k == "MODEL") {
      if (v == "BLACKOIL" || v == "BO") {
        d.model = (d.ncomp == 2) ? Model::BlackOil2P : Model::BlackOil3P;
      } else if (v == "COMPOSITIONAL" || v == "COMP") {
        d.model = Model::Compositional;
      }
    }
    // Unknown scalars are ignored (forward compatibility, OPM behaviour).
  };

  auto set_scalar = [&](const std::string& k, const std::string& v) {
    auto S = [&](Real& dst) { dst = toR(v); };
    auto I = [&](int& dst) { dst = toI(v); };
    if      (k == "NX") I(d.nx);      else if (k == "NY") I(d.ny);
    else if (k == "NZ") I(d.nz);      else if (k == "DX") S(d.dx);
    else if (k == "DY") S(d.dy);      else if (k == "DZ") S(d.dz);
    else if (k == "KZMULT") S(d.kz_mult);
    else if (k == "PERMMIN") S(d.perm_min); else if (k == "PERMMAX") S(d.perm_max);
    else if (k == "POROMIN") S(d.poro_min); else if (k == "POROMAX") S(d.poro_max);
    else if (k == "UW") S(d.uw); else if (k == "UO") S(d.uo); else if (k == "UG") S(d.ug);
    else if (k == "BW") S(d.bw); else if (k == "BO") S(d.bo); else if (k == "BG") S(d.bg);
    else if (k == "RHOW") S(d.rhow); else if (k == "RHOO") S(d.rhoo); else if (k == "RHOG") S(d.rhog);
    else if (k == "CFW") S(d.cfw); else if (k == "CFO") S(d.cfo); else if (k == "CT") S(d.ct);
    else if (k == "SWI") S(d.swi); else if (k == "SWR") S(d.swr);
    else if (k == "PINIT") S(d.p_init); else if (k == "PBUB") S(d.p_bub);
    else if (k == "PATM") S(d.p_atm);
    else if (k == "RS") S(d.rs_const);
    else if (k == "VAPOIL") I(d.vapoil);
    else if (k == "PVTCONST") I(d.pvt_const);
    else if (k == "PVTTABLES") I(d.pvt_tables);
    else if (k == "SWINIT") S(d.sw_init); else if (k == "SGINIT") S(d.sg_init);
    else if (k == "SOINIT") S(d.so_init);
    else if (k == "GRAVITY") I(d.gravity);
    else if (k == "TEMP") S(d.temperature);
    else if (k == "ZFACTOR") S(d.z_factor);
    else if (k == "DARCY") S(d.darcy_const);
    else if (k == "GRAVGRAD") S(d.grav_grad);
    else if (k == "REFACTOR") S(d.re_factor);
    else if (k == "FDEPS") S(d.fd_eps);
    else if (k == "TRANSDROP") S(d.trans_drop_rel);
    else if (k == "NCOMP") { I(d.ncomp); }
    else if (k == "KRYLOV") {
      if (v == "GMRES" || v == "FGMRES" || v == "0") d.krylov = 0;
      else if (v == "BICGSTAB" || v == "BICG" || v == "1") d.krylov = 1;
    }
    else if (k == "LINTOL") S(d.lin_tol);
    else if (k == "LINMAXIT") I(d.lin_maxit);
    else if (k == "RESTART") I(d.gmres_restart);
    else if (k == "ORTHOPASSES") I(d.ortho_passes);
    else if (k == "CPRW") I(d.use_cprw);
    else if (k == "AMGREUSE") I(d.amg_reuse);
    else if (k == "AMGMAXCOARSE") I(d.amg_max_coarse);
    else if (k == "AMGPRINT") I(d.amg_print_level);
    else if (k == "AMGCOARSERELAX") I(d.amg_coarse_relax);
    else if (k == "LINSOLVER" || k == "AMGLIB") {
      if (v == "AMGX" || v == "amgx" || v == "1") d.backend = 1;
      else if (v == "HYPRE" || v == "hypre" || v == "0") d.backend = 0;
      else if (v == "GINKGO" || v == "ginkgo" || v == "Ginkgo" || v == "2")
        d.backend = 2;
      else
        throw std::runtime_error(
            "LINSOLVER must be AMGX, HYPRE or GINKGO, got '" + v + "'");
    }
    else if (k == "GINKGOPREC") {
      // 16 / HALF / FP16, or 32 / SINGLE / FP32.  Anything else is an error:
      // a typo that silently ran FP32 would be reported as an FP16 result.
      if (v == "16" || v == "HALF" || v == "half" || v == "FP16" || v == "fp16")
        d.ginkgo_precision = 16;
      else if (v == "32" || v == "SINGLE" || v == "single" || v == "FP32" ||
               v == "fp32")
        d.ginkgo_precision = 32;
      else
        throw std::runtime_error("GINKGOPREC must be 16 or 32, got '" + v + "'");
    }
    else if (k == "PRECOND") {
      if (v == "MGR" || v == "1") d.use_mgr = 1;
      else if (v == "CPR" || v == "0") d.use_mgr = 0;
    }
    else if (k == "MGRRELAX") I(d.mgr_relax_type);
    else if (k == "MGRSWEEPS") I(d.mgr_relax_sweeps);
    else if (k == "MGRINTERP") I(d.mgr_interp_type);
    else if (k == "MGRSMOOTH") I(d.mgr_global_smooth);
    else if (k == "MGRSMOOTHITS") I(d.mgr_global_sweeps);
    else if (k == "AMGFAST") I(d.amg_fast);
    else if (k == "AMGSEQTHRESH") I(d.amg_seq_threshold);
    else if (k == "AMGNONGALERKIN") S(d.amg_nongalerkin_tol);
    else if (k == "AGGLEVELS") I(d.agg_num_levels);
    else if (k == "AGGTRUNC") S(d.agg_trunc_factor);
    else if (k == "AGGPMAX") I(d.agg_pmax);
    else if (k == "AMGREUSEGROWTH") S(d.amg_reuse_growth);
    else if (k == "APPLEYARD") I(d.appleyard);
    else if (k == "USEENDSCALE") I(d.use_endscale);
    else if (k == "PROPSARRAYS") I(d.props_arrays);
    else if (k == "FIELDSATFUNC") I(d.field_satfunc);
    // Reviewer-study controls (see deck.hpp for what each one does).
    else if (k == "CUDAGRAPHMINREUSE") S(d.graph_min_reuse);
    else if (k == "CUDAGRAPHGUARDAFTER") I(d.graph_guard_after);
    else if (k == "CPRDECOUPLE") {
      if (v == "DRS" || v == "1") d.cpr_decouple = 1;
      else if (v == "SUM" || v == "ROWSUM" || v == "2") d.cpr_decouple = 2;
      else d.cpr_decouple = 0;                    // QUASI, 0, anything else
    }
    else if (k == "DRSEPS") S(d.drs_eps);
    else if (k == "DTSEQFILE") d.dt_seq_file = v;
    // Anything not named above is a solver, timestep, well or schedule
    // control; see set_scalar_ctl.
    else set_scalar_ctl(k, v);
  };

  std::vector<std::string> list_keys = {"SWOF",  "SWOW",     "SGOF", "SWOG",
                                        "WELLS", "WELSPECS", "ZMF",  "TCRIT",
                                        "PCRIT", "ACF",      "VMOL"};

  while (std::getline(f, raw)) {
    std::string line = strip_comment(raw);
    if (line.empty()) continue;

    // End-of-record-list terminator.
    if (line == "/") {
      flush_records(kw, recs);
      kw.clear();
      continue;
    }

    // A bare keyword on its own line starts a section.
    auto tk = tokens(line);
    if (is_keyword(tk[0]) && tk.size() == 1) {
      flush_records(kw, recs);
      // METRIC and FIELD are unit-system switches, not section headers, and
      // they arrive as a bare keyword on their own line.  Handled here
      // because the branch below would otherwise take any lone keyword as the
      // start of a record list and swallow it -- which is what happened, and
      // the deck simulated in metric numbers read as field ones.
      if (tk[0] == "METRIC") { d.metric_input = 1; kw.clear(); continue; }
      if (tk[0] == "FIELD")  { d.metric_input = 0; kw.clear(); continue; }
      kw = tk[0];
      // Normalise NCOMP-dependent model default after RUNSPEC read.
      continue;
    }

    // A record inside a list keyword.
    if (std::find(list_keys.begin(), list_keys.end(), kw) != list_keys.end()) {
      // records may end with a trailing slash on the same line
      std::string rec = line;
      bool ends = false;
      if (!rec.empty() && rec.back() == '/') {
        rec.pop_back();
        ends = true;
      }
      recs.push_back(rec);
      if (ends) {
        flush_records(kw, recs);
        kw.clear();
      }
      continue;
    }

    // WELLPLOT names a LIST, so it cannot go through the scalar handler --
    // that takes tk[1] and drops the rest, which would silently reduce
    // "WELLPLOT WOPR WWCT WGOR" to WOPR alone.
    if (!tk.empty() && tk[0] == "WELLPLOT") {
      d.well_plot.clear();
      for (size_t q = 1; q < tk.size(); ++q) {
        std::string v = tk[q];
        if (v == "=") continue;
        for (size_t c = 0; c < v.size(); ++c)
          v[c] = (char)std::toupper((unsigned char)v[c]);
        if (!v.empty()) d.well_plot.push_back(v);
      }
      continue;
    }

    // DUMPOP step [newton]: two numbers, so not the scalar handler either.
    if (!tk.empty() && tk[0] == "DUMPOP") {
      if (tk.size() >= 2) d.dump_op_step = toI(tk[1]);
      if (tk.size() >= 3) d.dump_op_newton = toI(tk[2]);
      continue;
    }

    // Otherwise: KEY value  (scalar assignment) — accept `KEY = value` too.
    if (tk.size() >= 2) {
      std::string key = tk[0];
      std::string val = tk[1];
      if (val == "=" && tk.size() >= 3) val = tk[2];
      // strip quotes
      if (!val.empty() && (val.front() == '\'' || val.front() == '"'))
        val = val.substr(1, val.size() - 2);
      set_scalar(key, val);
    }
  }
  flush_records(kw, recs);

  // Post-conditions / defaults.
  if (d.model == Model::Compositional && d.comp_z.empty())
    throw std::runtime_error("COMPOSITIONAL model requires a ZMF feed record");
  if (d.trans_drop_rel < 0)
    throw std::runtime_error("TRANSDROP must be non-negative");
  if (d.model != Model::Compositional) d.ncomp = block_size(d.model);
  if (d.metric_input) to_field_units(d);
  return d;
}

void print_deck_summary(const Deck& d) {
  const char* mname = d.model == Model::Compositional ? "COMPOSITIONAL"
                      : d.model == Model::BlackOil2P  ? "BLACK-OIL (2-phase)"
                                                      : "BLACK-OIL (3-phase)";
  const char* kname = d.krylov == 1 ? "BiCGSTAB" : "FGMRES";
  std::cout << "======================================================\n"
            << "  NVRS-CUDA  —  " << d.title << "\n"
            << "------------------------------------------------------\n"
            << "  Model        : " << mname << "\n";

  // With a field deck the grid, the cell sizes and the wells all come from
  // that file, and this deck's NX/NY/NZ and well list are placeholders the
  // parser requires.  Printing them here states three things that are not
  // true of the run -- a 1 x 1 x 1 grid of one cell with no wells -- and the
  // reader has no way to tell that from a real description.  The corner-point
  // grid and the schedule print their own summaries immediately below.
  if (!d.field_deck.empty()) {
    std::cout << "  Grid         : from " << d.field_deck << "\n"
              << "  Wells        : from that deck's schedule\n"
              << "  Trans drop   : " << d.trans_drop_rel
              << " x maximum (0 keeps all positive connections)\n";
  } else {
    std::cout << "  Grid         : " << d.nx << " x " << d.ny << " x " << d.nz
              << "  (" << d.n_cells() << " cells)\n"
              << "  Cell size    : " << d.dx << " x " << d.dy << " x " << d.dz
              << " ft\n"
              << "  Wells        : " << d.n_inj() << " injectors, "
              << d.n_prod() << " producers\n";
  }

  std::cout << "  Solver       : CPR  +  " << kname << "  ["
            << (d.mixed_precision ? "mixed FP32 AMG" : "FP64 AMG")
            << (d.emulate_fp64 == 1 ? ", df64-emulated operator"
                : d.emulate_fp64 == 2 ? ", FP32-stored operator (FP64 master)"
                                      : "")
            << "]\n"
            << "  Time         : 0 → " << d.t_end << " days, report every "
            << d.dt_report << " days\n"
            << "======================================================\n";
}

// ---------------------------------------------------------------------------
// METRIC run decks
// ---------------------------------------------------------------------------
//
// The solver is field-units throughout, so a metric deck is converted once
// here rather than at every point of use.  What does NOT convert is as
// important as what does:
//
//   permeability   mD in both systems
//   viscosity      cP in both systems
//   formation vol  rm3/sm3 and rb/stb are numerically the same
//   time           days in both
//   saturations, relative permeabilities, mole fractions -- dimensionless
//
// and the solver's own constants -- DARCY, GRAVGRAD, REFACTOR -- are field
// constants by definition and are applied after this, so they stay put.
void to_field_units(Deck& d) {
  if (!d.metric_input) return;
  d.metric_input = 0;                    // idempotent: only ever once
  d.was_metric = 1;                      // remembered, so the run can say so

  constexpr Real kMToFt    = 3.280839895;
  constexpr Real kBarToPsi = 14.503773773;
  constexpr Real kM3ToBbl  = 6.289810770;
  constexpr Real kM3ToFt3  = 35.31466672;
  // sm3/sm3 -> scf/stb: gas in ft3 over liquid in bbl.
  constexpr Real kRsMToF   = kM3ToFt3 / kM3ToBbl;

  auto len   = [&](Real& x) { x *= kMToFt; };
  auto press = [&](Real& x) { x *= kBarToPsi; };
  auto comp  = [&](Real& x) { x /= kBarToPsi; };   // 1/bar -> 1/psi

  // Geometry.
  len(d.dx); len(d.dy); len(d.dz);

  // Pressures, and the Newton limit that is one.
  press(d.p_init); press(d.p_bub); press(d.p_atm);
  press(d.prod_bhp); press(d.inj_bhp); press(d.prod_draw_max);
  press(d.dp_max);

  // Compressibilities are reciprocal pressures and go the other way.
  comp(d.cfw); comp(d.cfo); comp(d.ct);

  // Density: METRIC states kg/m3, the solver wants g/cc.
  d.rhow /= 1000.0; d.rhoo /= 1000.0; d.rhog /= 1000.0;

  // Solution gas-oil ratio.
  d.rs_const *= kRsMToF;

  // Wells: radius is a length, the flowing pressure is a pressure, and the
  // rate is a surface volume.  A rate of -1 is the sentinel for "no rate
  // target, use BHP" and must survive unscaled.
  for (WellSpec& w : d.wells) {
    len(w.rw);
    press(w.pwf);
    if (w.rate > 0) w.rate *= kM3ToBbl;
  }

  // Capillary pressure columns of the saturation tables.
  for (Real& v : d.swof_pcow) press(v);
  for (Real& v : d.sgof_pcog) press(v);

  // Compositional: critical pressures, and temperatures from kelvin to
  // rankine.  Critical volume is a molar volume.
  for (Real& v : d.comp_Pc) press(v);
  for (Real& v : d.comp_Tc) v *= 1.8;
  for (Real& v : d.comp_vmL) v *= kM3ToFt3 / 1000.0;   // m3/kmol -> ft3/lbmol
  d.temperature *= 1.8;

  // Equilibration gradients, if the run deck states them: bar/m -> psi/ft.
  d.grad_oil   *= kBarToPsi / kMToFt;
  d.grad_water *= kBarToPsi / kMToFt;
  d.grad_gas   *= kBarToPsi / kMToFt;
}

}  // namespace nvrs
