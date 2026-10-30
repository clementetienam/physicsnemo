// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// physics_host.cu — host-side reservoir setup, sparsity/topology construction,
// Newton update and PVT refresh shared by the black-oil and compositional
// models.  Generates rock fields, uploads rel-perm tables, places wells with a
// Peaceman well index, and builds the 7-point block-CSR sparsity (coupled
// operator + pressure sub-block + per-cell block-diagonal layout).
#include "physics.hpp"
#include "halo.hpp"
#include "pvt.hpp"
#include "wellcell.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

namespace nvrs {

// --------------------------------------------------------------------------
// Device helpers
// --------------------------------------------------------------------------
__global__ static void k_relative_change(const Real* now, const Real* old,
                                         const Real* meaning,
                                         const Real* old_meaning,
                                         int bs, int cells, Real* out) {
  Real dn = 0, nn = 0;
  for (int c = threadIdx.x; c < cells; c += blockDim.x) {
    const size_t q = (size_t)bs * c;
    const Real dp = now[q] - old[q];
    dn += dp * dp;
    nn += now[q] * now[q];
    if (bs > 1) {
      const Real sw = now[q + 1], sw0 = old[q + 1];
      Real sg = 0, sg0 = 0;
      if (bs == 3) {
        const Real k = meaning ? meaning[c] : (Real)0;
        const Real k0 = old_meaning ? old_meaning[c] : (Real)0;
        sg = k > (Real)1.5 ? (Real)1 - sw
             : k > (Real)0.5 ? (Real)0 : now[q + 2];
        sg0 = k0 > (Real)1.5 ? (Real)1 - sw0
              : k0 > (Real)0.5 ? (Real)0 : old[q + 2];
      }
      const Real so = (Real)1 - sw - sg;
      const Real so0 = (Real)1 - sw0 - sg0;
      const Real dsw = sw - sw0, dsg = sg - sg0, dso = so - so0;
      dn += dsw * dsw + dsg * dsg + dso * dso;
      nn += sw * sw + sg * sg + so * so;
    }
  }
  __shared__ Real sd[256], sn[256];
  sd[threadIdx.x] = dn;
  sn[threadIdx.x] = nn;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) {
      sd[threadIdx.x] += sd[threadIdx.x + stride];
      sn[threadIdx.x] += sn[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[0] = sd[0];
    out[1] = sn[0];
  }
}

// Newton update with per-cell step limiting (the "Appleyard chop" every
// commercial simulator applies, and the analogue of the CuPy reference's
// alpha-halving line search).  A fully-implicit Newton step early in a report
// step routinely asks for pressure changes of thousands of psi and saturation
// changes far outside [0,1]; taking them literally walks the state into a
// region where the PVT correlations and rel-perm tables are meaningless and the
// iteration never recovers.  Clipping the *update* keeps every iterate physical
// while still following the Newton direction.
__global__ static void k_newton_update(Real* u, const Real* du, Real damp, int bs,
                                       int cells, Real swi, Real swr,
                                       Real dp_max, Real dp_max_rel,
                                       Real ds_max, int is_comp,
                                       int appleyard, Real* clamp_dso,
                                       const Real* gas_primary) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cells) return;
  auto clip = [](Real v, Real lim) { return fmin(fmax(v, -lim), lim); };

  const Real p0 = u[bs * c];
  const Real dp_lim = dp_max_rel > 0 ? dp_max_rel * fabs(p0) : dp_max;
  u[bs * c] = p0 + clip(damp * du[bs * c], dp_lim);

  if (is_comp) {
    // Compositional unknowns are overall molar densities N_1..N_nc, not
    // saturations.  They are bounded below by zero and not above by one, and
    // there are bs-1 of them.  The black-oil branch below would clamp N_1 into
    // [Swi, 1] as though it were a water saturation and never touch N_2..N_nc
    // at all, so a compositional Newton could only ever move p and N_1.
    for (int a = 1; a < bs; ++a) {
      const Real N = u[bs * c + a] + damp * du[bs * c + a];
      u[bs * c + a] = fmax(N, 0.0);
    }
    return;
  }

  const Real Sw0 = u[bs * c + 1];
  // The update Newton actually asked for, before any of the limiting below.
  // Its only use is the clamp diagnostic: So_raw - So_clamped is oil that the
  // limiter moved and no equation accounts for.
  const Real Sw_raw = Sw0 + damp * du[bs * c + 1];
  const Real Sg_raw =
      (bs == 3) ? u[bs * c + 2] + damp * du[bs * c + 2] : (Real)0;
  Real dsw = damp * du[bs * c + 1];
  Real dsg = (Real)0;
  const bool sg_primary = bs == 3 && gas_primary &&
                          gas_primary[c] < (Real)0.5;
  if (sg_primary) {
    dsg = damp * du[bs * c + 2];
    // OPM's saturation limiter uses one alpha for Sw, Sg and the dependent
    // So. Independent component clipping permits |dSo| up to 2*DSMAX and is
    // especially damaging at phase boundaries.
    const Real dso = -dsw - dsg;
    const Real dmax = fmax(fabs(dsw), fmax(fabs(dsg), fabs(dso)));
    if (dmax > ds_max) {
      const Real alpha = ds_max / dmax;
      dsw *= alpha;
      dsg *= alpha;
    }
  } else {
    dsw = clip(dsw, ds_max);
  }
  Real Sw = Sw0 + dsw;
  if (appleyard) {
    // Modified Appleyard chop (Appleyard & Cheshire; ECLIPSE and OPM carry
    // the same rule).  The relperm endpoints Swc = swi and Smax = 1 - swr are
    // where a phase's mobility switches on or off, and an update that CROSSES
    // one flips the cell into a different mobility regime in a single leap:
    // Newton then oscillates across the endpoint, the per-iteration clip
    // fights it, and the timestep chops.  Truncating such an update to land
    // just past the endpoint lets the regime change happen over two
    // well-behaved iterations instead.  Plain magnitude clipping cannot do
    // this: it is blind to where the endpoints are.
    // STRICT inequalities, so the truncation fires only in the iteration
    // that crosses the endpoint: a cell parked exactly at swc + eps by the
    // chop must be free to move in the next iteration.  With <= instead,
    // and SPE10 initialised at connate water everywhere, every invaded cell
    // was re-trapped at +eps per iteration and no timestep could converge.
    const Real eps = (Real)1e-3;
    const Real swc = swi, smax = (Real)1.0 - swr;
    if (Sw0 < swc + eps && Sw > swc + eps) Sw = swc + eps;
    else if (Sw0 > swc + eps && Sw < swc) Sw = swc;
    if (Sw0 > smax - eps && Sw < smax - eps) Sw = smax - eps;
    else if (Sw0 < smax - eps && Sw > smax) Sw = smax;
  }
  // SWI is the relperm-table endpoint, not a primary-variable bound. Flow's
  // chopAndNormalizeSaturations() projects physical saturations to [0,1]; it
  // permits a Newton iterate below connate water while krw remains clamped to
  // zero. Pinning Sw to SWI leaves an equal-and-opposite oil/water residual
  // that no subsequent linear solve can remove.
  Sw = fmin(fmax(Sw, (Real)0), (Real)1);
  u[bs * c + 1] = Sw;
  if (bs == 3) {
    const bool rs_primary = gas_primary && gas_primary[c] > (Real)0.5;
    if (rs_primary) {
      // Rs is a composition ratio, not a saturation: do not apply DSMAX or
      // [0,1-Sw].  Like OPM, only prevent a Newton update from making it
      // negative; the appearance test below may legitimately overshoot RsSat.
      u[bs * c + 2] = fmax((Real)0,
                            u[bs * c + 2] + damp * du[bs * c + 2]);
      if (clamp_dso) clamp_dso[c] += Sw_raw - Sw;
      return;
    }
    Real Sg = u[bs * c + 2] +
              (sg_primary ? dsg : clip(damp * du[bs * c + 2], ds_max));
    if (appleyard && !gas_primary) {
      // Gas appears and disappears at Sg = 0; the same truncation applies.
      const Real eps = (Real)1e-3;
      const Real Sg0 = u[bs * c + 2];
      if (Sg0 < eps && Sg > eps) Sg = eps;
      else if (Sg0 > eps && Sg < (Real)0.0) Sg = (Real)0.0;
    }
    // Switching needs the negative iterate as its disappearance signal.  OPM
    // adapts primary meaning before saturation normalization; applying the
    // legacy Appleyard endpoint chop first would turn every negative trial
    // into exactly zero and make Sg -> Rs impossible.
    Sg = gas_primary ? Sg
                     : fmin(fmax(Sg, 0.0), 1.0 - Sw);
    u[bs * c + 2] = Sg;
  }
  // So = 1 - Sw - Sg, so the oil the limiter moved is exactly the amount by
  // which it moved the other two.  One thread owns cell c, so no atomic.
  if (clamp_dso) {
    const Real Sg_f = (bs == 3) ? u[bs * c + 2] : (Real)0;
    clamp_dso[c] += (Sw_raw + Sg_raw) - (Sw + Sg_f);
  }
}

__global__ static void k_sum_pressure(const Real* u, int bs, int cells, Real* out) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < cells) atomicAdd(out, u[bs * c]);
}

__global__ static void k_split_df64(const Real* a, float* hi, float* lo, int nnz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nnz) return;
  float h = (float)a[i];
  hi[i] = h;
  lo[i] = (float)(a[i] - (double)h);
}

// --------------------------------------------------------------------------
// Reservoir construction
// --------------------------------------------------------------------------
Reservoir make_reservoir(const Deck& d, const Grid& g) {
  Reservoir r;
  r.grid = g;
  // On this path the grid IS the decomposition, so these simply mirror it.
  r.mpi_rank = g.rank;
  r.mpi_nranks = g.nranks;
  r.model = d.model;
  r.ncomp = d.ncomp;
  r.bs = block_size(d.model, d.ncomp);
  r.kz_mult = d.kz_mult;
  r.uw = d.uw; r.uo = d.uo; r.ug = d.ug;
  r.bw = d.bw; r.bo = d.bo; r.bg = d.bg;
  r.pvt_const = d.pvt_const;
  r.rs = d.pvt_const ? d.rs_const : pvt::rs(d.p_bub, d.p_init);
  r.swi = d.swi; r.swr = d.swr; r.cfo = d.cfo;
  r.ct = d.ct;
  r.appleyard = d.appleyard;
  r.pv_switch = d.pv_switch;
  r.aim = d.aim; r.aim_cfl = d.aim_cfl;
  r.p_bub = d.p_bub; r.p_atm = d.p_atm;
  r.rhow = d.rhow; r.rhoo = d.rhoo; r.rhog = d.rhog;
  r.gravity = d.gravity;
  r.jacobian = d.jacobian;
  r.dp_max = d.dp_max; r.dp_max_rel = d.dp_max_rel; r.ds_max = d.ds_max;
  r.darcy_const = d.darcy_const;
  r.grav_grad = d.grav_grad;
  r.re_factor = d.re_factor;
  r.fd_eps = d.fd_eps;
  r.temperature = d.temperature;
  r.z_factor = d.z_factor;

  const int cells = g.n_local();
  const int cells_ext = g.n_local_ext();
  const int nx = g.nx, ny = g.ny;

  // The halo staging buffers must cover the widest vector we ever exchange.
  halo_init(g, r.bs);

  // --- rock fields ---
  // If PERMFILE is given, read a flat global permeability list (one value per
  // global cell, natural order) and slice this rank's K-slab out of it; the
  // porosity is mapped from permeability by the deck's [poro_min,poro_max]
  // range.  Otherwise fall back to a deterministic smooth field so the tool
  // runs with no external data (never a physical assumption — just a stand-in).
  std::vector<Real> perm(cells), poro(cells);
  std::vector<Real> gperm;
  auto read_field = [&](const std::string& path, const char* kw,
                        std::vector<Real>& out) {
    std::ifstream pf(path);
    if (!pf)
      throw std::runtime_error(std::string("cannot open ") + kw + ": " + path);
    Real v;
    while (pf >> v) out.push_back(v);
    if ((int)out.size() < g.n_global())
      throw std::runtime_error(std::string(kw) +
                               " has fewer values than grid cells");
  };
  if (!d.perm_path.empty()) read_field(d.perm_path, "PERMFILE", gperm);
  // Optional independent porosity and vertical permeability, same format and
  // order as PERMFILE.  When POROFILE is absent the porosity is the linear
  // map of permeability onto [poro_min, poro_max]; when PERMZFILE is absent,
  // Kz = kz_mult * Kx.  POROMIN doubles as a floor under a read porosity:
  // SPE10's field carries near-zero values that would otherwise zero the
  // accumulation term of those cells.
  std::vector<Real> gporo, gpermz;
  if (!d.poro_path.empty()) read_field(d.poro_path, "POROFILE", gporo);
  if (!d.permz_path.empty()) read_field(d.permz_path, "PERMZFILE", gpermz);
  std::vector<Real> permz(gpermz.empty() ? 0 : cells);
  Real kmin = d.perm_min, kmax = d.perm_max;
  Real gkmin = 1e30, gkmax = -1e30;
  if (!gperm.empty()) {
    for (Real v : gperm) { gkmin = std::min(gkmin, v); gkmax = std::max(gkmax, v); }
  }
  // The rock is indexed by GLOBAL position, so the field a cell gets does not
  // depend on how the domain was cut.  PERMFILE is in natural global order.
  for (int kl = 0; kl < g.nzl; ++kl)
    for (int jl = 0; jl < g.nyl; ++jl)
      for (int il = 0; il < g.nxl; ++il) {
        int c = g.lid(il, jl, kl);
        int ig = g.i0 + il, jg = g.j0 + jl, kg = g.k0 + kl;
        if (!gperm.empty()) {
          int gc = (kg * ny + jg) * nx + ig;
          perm[c] = gperm[gc];
          if (!gporo.empty())
            poro[c] = std::max(gporo[gc], d.poro_min);
          else {
            Real tt = (gkmax > gkmin) ? (gperm[gc] - gkmin) / (gkmax - gkmin) : 0.5;
            poro[c] = d.poro_min + (d.poro_max - d.poro_min) * tt;
          }
          if (!gpermz.empty()) permz[c] = gpermz[gc];
        } else {
          Real s = std::sin(0.173 * (ig + 1)) + std::cos(0.119 * (jg + 2)) +
                   std::sin(0.071 * (kg + 3));
          Real t = 0.5 * (s + 3.0) / 3.0;               // → [0,1]
          perm[c] = kmin + (kmax - kmin) * t;
          poro[c] = d.poro_min + (d.poro_max - d.poro_min) * t;
        }
      }

  auto up = [](Real** dst, const std::vector<Real>& h) {
    NVRS_CHECK_CUDA(cudaMalloc(dst, sizeof(Real) * h.size()));
    NVRS_CHECK_CUDA(cudaMemcpy(*dst, h.data(), sizeof(Real) * h.size(),
                               cudaMemcpyHostToDevice));
  };
  // Cell-blocked fields are allocated over the *extended* index space so the
  // ghost layers have somewhere to land; only the owned prefix is initialised
  // here, the rest is filled by halo_exchange.
  auto up_ext = [&](Real** dst, const std::vector<Real>& h, int bs) {
    const size_t next = (size_t)bs * (size_t)cells_ext;
    NVRS_CHECK_CUDA(cudaMalloc(dst, sizeof(Real) * next));
    NVRS_CHECK_CUDA(cudaMemset(*dst, 0, sizeof(Real) * next));
    NVRS_CHECK_CUDA(cudaMemcpy(*dst, h.data(), sizeof(Real) * h.size(),
                               cudaMemcpyHostToDevice));
  };
  up_ext(&r.perm, perm, 1);
  up_ext(&r.poro, poro, 1);
  // Transmissibility across the slab seam needs the neighbour's permeability.
  halo_exchange(r.perm, 1, g);
  halo_exchange(r.poro, 1, g);
  if (!permz.empty()) {
    up_ext(&r.permz, permz, 1);
    halo_exchange(r.permz, 1, g);
  }
  // AIM implicitness flags, allocated unconditionally so the assembly
  // kernels can read them without a null branch: all-ones = plain FIM,
  // which is the default and the exact previous behaviour.
  {
    std::vector<Real> fl(cells, 1.0);
    up_ext(&r.aimflag, fl, 1);
    halo_exchange(r.aimflag, 1, g);
    std::vector<Real> zr(cells, 0.0);
    up_ext(&r.aim_rate, zr, 1);
  }

  // --- global K index of every local cell (interior + ghost) ---
  {
    std::vector<int> kk(cells_ext);
    for (int c = 0; c < cells_ext; ++c) kk[c] = g.global_k(c);
    NVRS_CHECK_CUDA(cudaMalloc(&r.cellk, sizeof(int) * cells_ext));
    NVRS_CHECK_CUDA(cudaMemcpy(r.cellk, kk.data(), sizeof(int) * cells_ext,
                               cudaMemcpyHostToDevice));
  }

  // --- initial primary variables ---
  std::vector<Real> u(r.bs * cells);
  if (d.model == Model::Compositional) {
    // Primary unknowns are [p, N_1..N_nc] with N_i the moles of component i per
    // unit *pore* volume.  Seed them from the deck feed ZMF at the initial
    // pressure using the same closure the device flash uses, scaled so the
    // fluid exactly fills the pore volume:
    //     v_mix = (1-β)/ρ_L + β/ρ_V   [ft³/lbmol],   N_tot = 1 / v_mix,
    // which makes the volume-balance residual vanish at t = 0.  Leaving these
    // at zero (the previous behaviour) starts Newton from an empty reservoir.
    const int nc = d.ncomp;
    if (nc < 1 || nc > kMaxComponents)
      throw std::runtime_error(
          "NCOMP must be between 1 and " + std::to_string(kMaxComponents) +
          " (the compositional kernels use fixed-size per-thread storage)");
    if ((int)d.comp_z.size() < nc)
      throw std::runtime_error("ZMF must supply NCOMP mole fractions");
    if ((int)d.comp_Tc.size() < nc || (int)d.comp_Pc.size() < nc ||
        (int)d.comp_acf.size() < nc || (int)d.comp_vmL.size() < nc)
      throw std::runtime_error(
          "COMPOSITIONAL model needs TCRIT/PCRIT/ACF/VMOL with NCOMP entries each");
    std::vector<Real> z(d.comp_z.begin(), d.comp_z.begin() + nc);
    Real zsum = 0;
    for (Real v : z) zsum += v;
    if (zsum <= 0) throw std::runtime_error("ZMF mole fractions must sum to > 0");
    for (Real& v : z) v /= zsum;                       // normalise the feed

    const Real T = d.temperature, p0 = d.p_init;
    std::vector<Real> K(nc);
    for (int i = 0; i < nc; ++i)
      K[i] = (d.comp_Pc[i] / p0) *
             std::exp(5.373 * (1.0 + d.comp_acf[i]) * (1.0 - d.comp_Tc[i] / T));

    // Rachford-Rice by bisection (same scheme as the device kernel).
    auto gRR = [&](Real b) {
      Real s = 0;
      for (int i = 0; i < nc; ++i) s += z[i] * (K[i] - 1.0) / (1.0 + b * (K[i] - 1.0));
      return s;
    };
    Real beta;
    if (gRR(0.0) < 0)      beta = 0.0;                 // all liquid
    else if (gRR(1.0) > 0) beta = 1.0;                 // all vapour
    else {
      Real lo = 0.0, hi = 1.0;
      for (int it = 0; it < 60; ++it) {
        Real mid = 0.5 * (lo + hi);
        (gRR(mid) > 0 ? lo : hi) = mid;
      }
      beta = 0.5 * (lo + hi);
    }

    const Real Rgas = 10.7316;                          // psia·ft³/(lbmol·R)
    Real vmL = 0;
    for (int i = 0; i < nc; ++i)
      vmL += (z[i] / (1.0 + beta * (K[i] - 1.0))) * d.comp_vmL[i];
    Real rhoV = p0 / (d.z_factor * Rgas * T);
    Real rhoL = (vmL > 1e-12) ? 1.0 / vmL : 1.0;
    Real v_mix = (1.0 - beta) / rhoL + beta / rhoV;     // ft³ per lbmol of feed
    Real Ntot = (v_mix > 1e-30) ? 1.0 / v_mix : 1.0;

    for (int c = 0; c < cells; ++c) {
      u[r.bs * c + 0] = p0;
      for (int i = 0; i < nc; ++i) u[r.bs * c + 1 + i] = z[i] * Ntot;
    }
    // Injectors inject the feed composition; the assembler needs it per kernel.
    for (int i = 0; i < nc; ++i) r.comp_zinj[i] = z[i];
  } else {
    for (int c = 0; c < cells; ++c) {
      u[r.bs * c + 0] = d.p_init;
      u[r.bs * c + 1] = d.sw_init;
      if (r.bs == 3) u[r.bs * c + 2] = d.sg_init;
    }
  }
  up_ext(&r.u, u, r.bs);
  up_ext(&r.u_old, u, r.bs);
  // Clamp diagnostic, over the same extent the update kernel walks.
  NVRS_CHECK_CUDA(cudaMalloc(&r.clamp_dso, sizeof(Real) * cells_ext));
  NVRS_CHECK_CUDA(cudaMemset(r.clamp_dso, 0, sizeof(Real) * cells_ext));
  halo_exchange(r.u, r.bs, g);
  halo_exchange(r.u_old, r.bs, g);

  // --- rel-perm tables ---
  auto upload_tab = [](const std::vector<SatRow>& tab, const Real** x,
                       const Real** ko, const Real** kwg, int& n) {
    n = (int)tab.size();
    std::vector<Real> X(n), KO(n), KWG(n);
    for (int i = 0; i < n; ++i) { X[i] = tab[i][0]; KO[i] = tab[i][1]; KWG[i] = tab[i][2]; }
    Real *dx, *dko, *dkwg;
    NVRS_CHECK_CUDA(cudaMalloc(&dx, sizeof(Real) * n));
    NVRS_CHECK_CUDA(cudaMalloc(&dko, sizeof(Real) * n));
    NVRS_CHECK_CUDA(cudaMalloc(&dkwg, sizeof(Real) * n));
    cudaMemcpy(dx, X.data(), sizeof(Real) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dko, KO.data(), sizeof(Real) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dkwg, KWG.data(), sizeof(Real) * n, cudaMemcpyHostToDevice);
    *x = dx; *ko = dko; *kwg = dkwg;
  };
  if (!d.swof.empty())
    upload_tab(d.swof, &r.tables.sw, &r.tables.krow, &r.tables.krw, r.tables.nsw);
  if (!d.sgof.empty())
    upload_tab(d.sgof, &r.tables.sg, &r.tables.krog, &r.tables.krg, r.tables.nsg);
  // The table's own endpoints, for ENDSCALE.  A structured deck carries no
  // endpoint arrays, so es.on stays 0 and relperm3 takes its unscaled branch;
  // these are filled anyway so the two build paths describe the same tables.
  {
    std::vector<Real> hsw, hkrw, hkrow, hsg, hkrg, hkrog;
    for (const auto& row : d.swof) {
      hsw.push_back(row[0]); hkrow.push_back(row[1]); hkrw.push_back(row[2]);
    }
    for (const auto& row : d.sgof) {
      hsg.push_back(row[0]); hkrog.push_back(row[1]); hkrg.push_back(row[2]);
    }
    set_table_endpoints(r.tables, hsw.data(), hkrw.data(), hkrow.data(),
                        (int)hsw.size(), hsg.data(), hkrg.data(),
                        hkrog.data(), (int)hsg.size());
  }
  // Capillary pressure, when the deck gave a fourth column with anything in
  // it.  All zeros is the same as absent and is treated as absent, so a deck
  // that pads its tables does not pay for a term that is identically zero.
  auto upload_pc = [&](const std::vector<Real>& col, const Real** dst) {
    if (col.empty()) return;
    bool any = false;
    for (Real v : col) if (v != (Real)0) { any = true; break; }
    if (!any) return;
    Real* dv = nullptr;
    NVRS_CHECK_CUDA(cudaMalloc(&dv, sizeof(Real) * col.size()));
    NVRS_CHECK_CUDA(cudaMemcpy(dv, col.data(), sizeof(Real) * col.size(),
                               cudaMemcpyHostToDevice));
    *dst = dv;
    r.tables.have_pc = 1;
  };
  if (!d.swof.empty() && d.swof_pcow.size() == d.swof.size())
    upload_pc(d.swof_pcow, &r.tables.pcow);
  if (!d.sgof.empty() && d.sgof_pcog.size() == d.sgof.size())
    upload_pc(d.sgof_pcog, &r.tables.pcog);

  // --- compositional EOS component properties (device arrays) ---
  if (d.model == Model::Compositional) {
    if ((int)d.comp_Tc.size() < d.ncomp || (int)d.comp_Pc.size() < d.ncomp ||
        (int)d.comp_acf.size() < d.ncomp || (int)d.comp_vmL.size() < d.ncomp)
      throw std::runtime_error(
          "COMPOSITIONAL model needs TCRIT/PCRIT/ACF/VMOL with NCOMP entries each");
    up(&r.comp_Tc, d.comp_Tc);
    up(&r.comp_Pc, d.comp_Pc);
    up(&r.comp_acf, d.comp_acf);
    up(&r.comp_vmL, d.comp_vmL);
  }

  // --- wells (per-cell WellCell with Peaceman well index) ---
  std::vector<WellCell> wc(cells);
  const Real Cd = d.darcy_const;
  const Real re = d.re_factor * d.dx;
  for (const auto& w : d.wells) {
    int i = w.i - 1, j = w.j - 1;                 // deck is 1-based
    if (i < 0 || i >= nx || j < 0 || j >= ny) continue;
    // Peaceman is only meaningful when the wellbore is inside the equivalent
    // drainage radius.  With r_w >= r_e the logarithm goes non-positive and the
    // well index changes sign, which silently turns producers into injectors:
    // pressure then runs away until the PVT correlations overflow and every
    // residual becomes NaN.  Refuse the deck instead.
    const Real denom = std::log(re / w.rw) + w.skin;
    if (!(denom > 0.0))
      throw std::runtime_error(
          "well '" + w.name + "': ln(re/rw) + skin = " + std::to_string(denom) +
          " must be > 0, but re = REFACTOR*DX = " + std::to_string(re) +
          " ft and rw = " + std::to_string(w.rw) +
          " ft. The wellbore radius must be well inside the equivalent "
          "drainage radius (a real rw is ~0.25-0.5 ft); increase DX/REFACTOR "
          "or fix RW in the WELLS record.");
    // Under a 3-D cut a well's areal column belongs to only some ranks, where
    // K-slabs gave every rank the whole (i,j) plane.  Ranks that do not own the
    // column contribute nothing to the well index sum, but they must still
    // enter the Allreduce below, so the test guards the loops and not the
    // collective.
    const bool mine = g.owns_column(i, j);
    const int il = i - g.i0, jl = j - g.j0;
    int klo = 0, khi = g.nzl - 1;
    // "last" means the bottom layer of the RESERVOIR, not of this rank's block;
    // only the rank owning the global bottom layer has that perforation.
    if (w.perf == "last") {
      klo = khi = g.nz - 1 - g.k0;
      if (klo < 0 || klo >= g.nzl) { klo = 0; khi = -1; }   // not on this rank
    }

    // A rate-controlled injector's RATE is the rate of the *well*, so it has to
    // be split across the perforations rather than applied in full to each one
    // — otherwise a well perforated in N layers injects N times its target.
    // The split is weighted by the well index, which is the standard
    // completion-productivity allocation.  The sum is taken over the *global*
    // perforation set so the split does not change with the rank count.
    Real wi_sum_local = 0.0;
    if (mine)
      for (int kl = klo; kl <= khi; ++kl) {
        int kg = g.k0 + kl;
        if (kg < 0 || kg >= g.nz) continue;
        int c = g.lid(il, jl, kl);
        wi_sum_local += 2.0 * units::kPi * perm[c] * d.dz / denom * Cd;
      }
    Real wi_sum = wi_sum_local;
    if (g.nranks > 1)
      MPI_Allreduce(MPI_IN_PLACE, &wi_sum, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
    if (wi_sum <= 0) wi_sum = 1.0;   // no perforation on any rank
    if (!mine) continue;

    for (int kl = klo; kl <= khi; ++kl) {
      int kg = g.k0 + kl;
      if (kg < 0 || kg >= g.nz) continue;
      int c = g.lid(il, jl, kl);
      Real kperm = perm[c];
      Real wi = 2.0 * units::kPi * kperm * d.dz / denom * Cd;
      if (w.type > 0) {
        wc[c].is_inj = 1;
        wc[c].qw_inj = w.rate * (wi / wi_sum);   // WI-weighted allocation
      } else {
        wc[c].is_prod = 1;
        wc[c].pwf = w.pwf;
      }
      wc[c].wi = wi;
    }
  }
  WellCell* d_wc = nullptr;
  NVRS_CHECK_CUDA(cudaMalloc(&d_wc, sizeof(WellCell) * cells));
  NVRS_CHECK_CUDA(cudaMemcpy(d_wc, wc.data(), sizeof(WellCell) * cells,
                             cudaMemcpyHostToDevice));
  r.wellcells = d_wc;
  r.wellcells_host = new WellCell[(size_t)cells];
  std::copy(wc.begin(), wc.end(), static_cast<WellCell*>(r.wellcells_host));

  return r;
}

void free_reservoir(Reservoir& r) {
  for (void* p : {(void*)r.perm, (void*)r.poro, (void*)r.permz,
                  (void*)r.aimflag, (void*)r.aim_rate, (void*)r.u,
                  (void*)r.u_old, (void*)r.gas_primary,
                  (void*)r.gas_primary_old,
                  (void*)r.nbr, (void*)r.bcol, (void*)r.pcolpos, (void*)r.tface,
                  (void*)r.cellk, (void*)r.comp_Tc, (void*)r.comp_Pc,
                  (void*)r.comp_acf, (void*)r.comp_vmL, r.wellcells,
                  (void*)r.well_update_cells, r.well_update_values,
                  (void*)r.well_state_cells, (void*)r.well_state_values,
                  r.assembly_pvt_cache,
                  (void*)r.well_region_partials,
                  (void*)r.assembly_residual_stats,
                  (void*)r.assembly_cnv_stats,
                  (void*)r.timestep_change_stats,
                  r.well_schur,
                  (void*)r.conn_ptr, (void*)r.conn_nbr, (void*)r.conn_bcol,
                  (void*)r.conn_pcol, (void*)r.self_bcol, (void*)r.self_pcol,
                  (void*)r.conn_trans, (void*)r.conn_dz,
                  (void*)r.conn_threshold,
                  (void*)r.cellvol, (void*)r.celldepth,
                  (void*)r.clamp_dso, (void*)r.sg_hy, (void*)r.sw_ow_hy,
                  (void*)r.so_max,
                  (void*)r.tables.pcow_scale})
    if (p) cudaFree(p);
  r.perm = r.poro = r.permz = r.aimflag = r.aim_rate = nullptr;
  r.u = r.u_old = r.gas_primary = r.gas_primary_old = nullptr;
  r.tface = r.clamp_dso = r.sg_hy = r.sw_ow_hy = nullptr;
  r.so_max = nullptr;
  r.tables.pcow_scale = nullptr;
  r.nbr = r.bcol = r.pcolpos = r.cellk = nullptr;
  r.conn_ptr = r.conn_nbr = r.conn_bcol = r.conn_pcol = nullptr;
  r.self_bcol = r.self_pcol = nullptr;
  r.conn_trans = r.conn_dz = r.conn_threshold = r.cellvol = r.celldepth = nullptr;
  r.nconn = r.max_degree = 0;
  r.comp_Tc = r.comp_Pc = r.comp_acf = r.comp_vmL = nullptr;
  delete[] static_cast<WellCell*>(r.wellcells_host);
  r.wellcells_host = nullptr;
  r.wellcells = nullptr;
  r.well_update_cells = nullptr;
  r.well_update_values = nullptr;
  r.well_update_capacity = 0;
  r.well_state_cells = nullptr;
  r.well_state_values = nullptr;
  r.well_state_capacity = 0;
  r.assembly_pvt_cache = nullptr;
  r.assembly_pvt_capacity = 0;
  r.assembly_pvt_ready = 0;
  r.well_region_partials = nullptr;
  r.well_region_capacity = 0;
  r.assembly_residual_stats = nullptr;
  r.assembly_cnv_stats = nullptr;
  r.timestep_change_stats = nullptr;
  r.well_schur = nullptr;
  r.well_schur_count = r.well_schur_capacity = r.well_schur_groups = 0;
  free_device_pvt(r.pvt);
}

// --------------------------------------------------------------------------
// Sparsity + topology + system allocation.
// --------------------------------------------------------------------------
void build_sparsity(const Reservoir& r, DeviceLinearSystem& sys) {
  const Grid& g = r.grid;
  const int nxl = g.nxl, nyl = g.nyl, nzl = g.nzl;
  const int cells = g.n_local();
  const int cells_ext = g.n_local_ext();
  const int bs = r.bs;
  sys.bs = bs;
  sys.cells = cells;
  sys.n = bs * cells;
  sys.cells_ext = cells_ext;
  sys.n_ext = bs * cells_ext;
  sys.cells_global = g.n_global();
  sys.row0 = g.row0();

  // Neighbour offsets: x-, x+, y-, y+, z-, z+, matching the Face enum.
  //
  // A neighbour that leaves this rank's block resolves to the ghost slab for
  // that face when another rank owns it, and is genuinely no-flow only at the
  // true reservoir boundary.  With a 3-D decomposition any of the six faces can
  // be a rank boundary, where the old K-slab code only had to consider two.
  auto neigh = [&](int il, int jl, int kl, int f) -> int {
    int ni = il, nj = jl, nk = kl;
    if (f == kFaceXlo) ni--; else if (f == kFaceXhi) ni++;
    else if (f == kFaceYlo) nj--; else if (f == kFaceYhi) nj++;
    else if (f == kFaceZlo) nk--; else nk++;
    // Off the end of the block: a ghost if a rank owns it, otherwise the
    // reservoir boundary.
    if (ni < 0)    return g.nbr[kFaceXlo] >= 0 ? g.gid_face(kFaceXlo, nj, nk) : -1;
    if (ni >= nxl) return g.nbr[kFaceXhi] >= 0 ? g.gid_face(kFaceXhi, nj, nk) : -1;
    if (nj < 0)    return g.nbr[kFaceYlo] >= 0 ? g.gid_face(kFaceYlo, ni, nk) : -1;
    if (nj >= nyl) return g.nbr[kFaceYhi] >= 0 ? g.gid_face(kFaceYhi, ni, nk) : -1;
    if (nk < 0)    return g.nbr[kFaceZlo] >= 0 ? g.gid_face(kFaceZlo, ni, nj) : -1;
    if (nk >= nzl) return g.nbr[kFaceZhi] >= 0 ? g.gid_face(kFaceZhi, ni, nj) : -1;
    return g.lid(ni, nj, nk);
  };

  // Copy perm (interior *and* ghost) to host to build face transmissibilities.
  std::vector<Real> perm(cells_ext);
  NVRS_CHECK_CUDA(cudaMemcpy(perm.data(), r.perm, sizeof(Real) * cells_ext,
                             cudaMemcpyDeviceToHost));
  // Per-cell Kz, when the deck supplied one (PERMZFILE); empty otherwise and
  // the vertical face falls back to Kz = kz_mult * Kx.
  std::vector<Real> permz;
  if (r.permz) {
    permz.resize(cells_ext);
    NVRS_CHECK_CUDA(cudaMemcpy(permz.data(), r.permz,
                               sizeof(Real) * cells_ext,
                               cudaMemcpyDeviceToHost));
  }
  const Real Cd = r.darcy_const;
  const Real Ax = g.dy * g.dz, Ay = g.dx * g.dz, Az = g.dx * g.dy;

  std::vector<int> nbr(7 * cells, -1), bcol(7 * cells, -1), pcolpos(7 * cells, -1);
  std::vector<Real> tface(6 * cells, 0.0);
  std::vector<int> h_rowptr(bs * cells + 1, 0), h_colind;
  std::vector<int> h_prowptr(cells + 1, 0), h_pcolind;

  for (int kl = 0; kl < nzl; ++kl)
    for (int j = 0; j < nyl; ++j)
      for (int i = 0; i < nxl; ++i) {
        int c = g.lid(i, j, kl);
        nbr[7 * c + 0] = c;
        std::vector<int> members = {c};
        for (int f = 0; f < 6; ++f) {
          int nc = neigh(i, j, kl, f);
          nbr[7 * c + 1 + f] = nc;
          if (nc >= 0) {
            members.push_back(nc);
            // harmonic transmissibility, directional perm
            Real kc = perm[c], kn = perm[nc];
            Real kh = 2.0 * kc * kn / (kc + kn + 1e-30);
            Real T;
            if (f < 2)      T = Cd * kh * Ax / g.dx;
            else if (f < 4) T = Cd * kh * Ay / g.dy;
            else if (permz.empty()) T = Cd * (r.kz_mult * kh) * Az / g.dz;
            else {
              Real kzc = permz[c], kzn = permz[nc];
              Real khz = 2.0 * kzc * kzn / (kzc + kzn + 1e-30);
              T = Cd * khz * Az / g.dz;
            }
            tface[6 * c + f] = T;
          }
        }
        std::sort(members.begin(), members.end());
        // scalar CSR rows (bs identical rows per cell).  `members` is sorted and
        // ghost indices exceed every interior index, so the column indices are
        // emitted in ascending order as cuSPARSE expects.
        int ncols_cells = (int)members.size();
        for (int a = 0; a < bs; ++a) {
          int row = bs * c + a;
          h_rowptr[row + 1] = ncols_cells * bs;
          for (int mc : members)
            for (int v = 0; v < bs; ++v) h_colind.push_back(bs * mc + v);
        }
        // Pressure CSR row.  HYPRE's IJ interface takes *global* column ids and
        // resolves off-process columns itself, so a ghost member contributes
        // its true global id and the AMG hierarchy spans the rank boundary.
        h_prowptr[c + 1] = ncols_cells;
        for (int mc : members) h_pcolind.push_back(g.global_cell(mc));
        // per-member block offsets
        for (int m = 0; m < 7; ++m) {
          int mc = nbr[7 * c + m];
          if (mc < 0) continue;
          auto it = std::lower_bound(members.begin(), members.end(), mc);
          int rank = (int)(it - members.begin());
          bcol[7 * c + m] = rank * bs;
          pcolpos[7 * c + m] = rank;
        }
      }

  // prefix sums
  for (int i = 0; i < bs * cells; ++i) h_rowptr[i + 1] += h_rowptr[i];
  for (int i = 0; i < cells; ++i) h_prowptr[i + 1] += h_prowptr[i];
  sys.nnz = (int)h_colind.size();
  sys.pnnz = (int)h_pcolind.size();

  // ---- the same connectivity, as a CSR adjacency -------------------------
  // Built from the arrays just filled, so the two descriptions cannot drift:
  // whatever the 7-point stencil says, this says the same thing in a form
  // that does not assume the count is six.  See physics.hpp for why.
  //
  // The depth difference carried per connection is what replaces the kernels'
  // "face index >= 4 means vertical" test.  Here it comes from the global K
  // index, so gravity is measured from the global datum exactly as the fixed
  // stencil did, and remains independent of how the grid was decomposed.
  std::vector<int> h_cptr(cells + 1, 0), h_cnbr, h_cbcol, h_cpcol;
  std::vector<Real> h_ctrans, h_cdz;
  std::vector<int> h_sbcol(cells, -1), h_spcol(cells, -1);
  h_cnbr.reserve((size_t)cells * 6);
  h_ctrans.reserve((size_t)cells * 6);
  h_cdz.reserve((size_t)cells * 6);
  h_cbcol.reserve((size_t)cells * 6);
  h_cpcol.reserve((size_t)cells * 6);

  int max_deg = 0;
  for (int c = 0; c < cells; ++c) {
    h_sbcol[c] = bcol[7 * c + 0];
    h_spcol[c] = pcolpos[7 * c + 0];
    const int kc = g.global_k(c);
    int deg = 0;
    for (int f = 0; f < 6; ++f) {
      const int nc = nbr[7 * c + 1 + f];
      if (nc < 0) continue;
      const Real T = tface[6 * c + f];
      if (!(T > 0)) continue;          // a no-flow face is not a connection
      h_cnbr.push_back(nc);
      h_ctrans.push_back(T);
      // P.dz_cell is r.grid.dz, and the kernels form depth as
      // (global_k + 0.5) * dz_cell, so this difference is exactly the one the
      // fixed stencil computed -- the +0.5 cancels.
      h_cdz.push_back((Real)(g.global_k(nc) - kc) * g.dz);
      h_cbcol.push_back(bcol[7 * c + 1 + f]);
      h_cpcol.push_back(pcolpos[7 * c + 1 + f]);
      ++deg;
    }
    h_cptr[c + 1] = h_cptr[c] + deg;
    max_deg = std::max(max_deg, deg);
  }

  auto up_i = [](int** dst, const std::vector<int>& h) {
    NVRS_CHECK_CUDA(cudaMalloc(dst, sizeof(int) * h.size()));
    NVRS_CHECK_CUDA(cudaMemcpy(*dst, h.data(), sizeof(int) * h.size(),
                               cudaMemcpyHostToDevice));
  };
  up_i(&sys.rowptr, h_rowptr);
  up_i(&sys.colind, h_colind);
  up_i(&sys.prowptr, h_prowptr);
  up_i(&sys.pcolind, h_pcolind);
  // topology arrays live on the Reservoir (const_cast: build-once fields)
  Reservoir& rr = const_cast<Reservoir&>(r);
  up_i(&rr.nbr, nbr);
  up_i(&rr.bcol, bcol);
  up_i(&rr.pcolpos, pcolpos);
  // The ZLINE stage 2 reads the vertical coupling blocks out of the coupled
  // CSR through the same topology; shared pointers, owned by the Reservoir.
  sys.nbr7 = rr.nbr;
  sys.bcol7 = rr.bcol;
  NVRS_CHECK_CUDA(cudaMalloc(&rr.tface, sizeof(Real) * tface.size()));
  NVRS_CHECK_CUDA(cudaMemcpy(rr.tface, tface.data(), sizeof(Real) * tface.size(),
                             cudaMemcpyHostToDevice));

  // the connection list
  auto up_r = [](Real** dst, const std::vector<Real>& h) {
    const size_t n = h.empty() ? 1 : h.size();
    NVRS_CHECK_CUDA(cudaMalloc(dst, sizeof(Real) * n));
    if (!h.empty())
      NVRS_CHECK_CUDA(cudaMemcpy(*dst, h.data(), sizeof(Real) * h.size(),
                                 cudaMemcpyHostToDevice));
  };
  up_i(&rr.conn_ptr, h_cptr);
  up_i(&rr.conn_nbr, h_cnbr);
  up_i(&rr.conn_bcol, h_cbcol);
  up_i(&rr.conn_pcol, h_cpcol);
  up_i(&rr.self_bcol, h_sbcol);
  up_i(&rr.self_pcol, h_spcol);
  up_r(&rr.conn_trans, h_ctrans);
  up_r(&rr.conn_dz, h_cdz);
  rr.nconn = (int)h_cnbr.size();
  rr.max_degree = max_deg;

  // system value arrays
  NVRS_CHECK_CUDA(cudaMalloc(&sys.aval64, sizeof(Real) * sys.nnz));
  NVRS_CHECK_CUDA(cudaMalloc(&sys.b, sizeof(Real) * sys.n));
  NVRS_CHECK_CUDA(cudaMalloc(&sys.pval, sizeof(PrecReal) * sys.pnnz));
  NVRS_CHECK_CUDA(cudaMalloc(&sys.dinv, sizeof(PrecReal) * (long long)cells * bs * bs));
  // Quasi-IMPES row weights used to build the pressure block; the CPR residual
  // restriction must apply the same weights (see linear_solver.hpp).
  NVRS_CHECK_CUDA(cudaMalloc(&sys.pweight, sizeof(PrecReal) * (long long)cells * bs));
}

// --------------------------------------------------------------------------
// Newton update + PVT refresh
// --------------------------------------------------------------------------
void newton_update(Reservoir& r, const Real* du, Real damp) {
  // Any property cache prepared by a control-only call describes the state
  // before this update and must never be consumed by the next assembly.
  r.assembly_pvt_ready = 0;
  int cells = r.grid.n_local();
  k_newton_update<<<grid1d(cells), 256>>>(r.u, du, damp, r.bs, cells, r.swi,
                                          r.swr, r.dp_max, r.dp_max_rel, r.ds_max,
                                          r.model == Model::Compositional,
                                          r.appleyard, r.clamp_dso,
                                          r.gas_primary);
  NVRS_CHECK_CUDA(cudaGetLastError());
  if (r.gas_primary) blackoil_pvswitch_adapt(r);
}

Real blackoil_relative_change(Reservoir& r, const Real* previous) {
  if (!r.timestep_change_stats)
    NVRS_CHECK_CUDA(cudaMalloc(&r.timestep_change_stats, 2 * sizeof(Real)));
  k_relative_change<<<1, 256>>>(r.u, previous, r.gas_primary,
                                r.gas_primary_old, r.bs, r.grid.n_local(),
                                r.timestep_change_stats);
  NVRS_CHECK_CUDA(cudaGetLastError());
  Real sums[2] = {0, 0};
  NVRS_CHECK_CUDA(cudaMemcpy(sums, r.timestep_change_stats, 2 * sizeof(Real),
                             cudaMemcpyDeviceToHost));
  if (r.mpi_nranks > 1)
    MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sums[1] > 0 ? sums[0] / sums[1] : (Real)0;
}

void refresh_pvt(Reservoir& r) {
  // PVTCONST: the deck fixed the fluid, so there is nothing to re-evaluate.
  // Returning before the Allreduce below is safe because every rank sees the
  // same deck and so takes the same branch.
  if (r.pvt_const) return;
  int cells = r.grid.n_local();
  Real* d_sum = nullptr;
  NVRS_CHECK_CUDA(cudaMalloc(&d_sum, sizeof(Real)));
  NVRS_CHECK_CUDA(cudaMemset(d_sum, 0, sizeof(Real)));
  k_sum_pressure<<<grid1d(cells), 256>>>(r.u, r.bs, cells, d_sum);
  Real sum = 0;
  NVRS_CHECK_CUDA(cudaMemcpy(&sum, d_sum, sizeof(Real), cudaMemcpyDeviceToHost));
  cudaFree(d_sum);
  long long gcells = cells;
  if (r.mpi_nranks > 1) {
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    long long lc = cells;
    MPI_Allreduce(&lc, &gcells, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  }
  Real pmean = sum / (Real)gcells;
  r.bo = pvt::bo(r.p_bub, r.p_atm, r.cfo, pmean);
  r.bg = pvt::bg(r.p_bub, r.p_atm, pmean) / 5.61458;
  r.rs = pvt::rs(r.p_bub, pmean);
  r.ug = pvt::mu_g(pmean);
}

void split_operator_df64(DeviceLinearSystem& sys) {
  if (!sys.aval_hi || !sys.aval_lo) return;
  k_split_df64<<<grid1d(sys.nnz), 256>>>(sys.aval64, sys.aval_hi, sys.aval_lo,
                                         sys.nnz);
  NVRS_CHECK_CUDA(cudaGetLastError());
}

}  // namespace nvrs
