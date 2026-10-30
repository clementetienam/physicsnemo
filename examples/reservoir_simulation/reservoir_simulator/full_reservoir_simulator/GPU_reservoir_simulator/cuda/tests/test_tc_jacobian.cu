// SPDX-License-Identifier: Apache-2.0
//
// test_tc_jacobian.cu -- the Tensor Core Jacobian against the analytic kernels.
//
// Each fill kernel in src/tc_assemble.inc claims that L_flux L_props equals the
// connection part of one analytic kernel.  This runs both on a small box with
// random state, random PVT slopes, capillary pressure, AIM flags and THPRES, so
// that every lane carries a term of comparable size, and compares:
//
//   neighbour blocks   product against analytic, entry by entry
//   self blocks        accumulation + FP64 factorised sum against analytic
//   self blocks        after the product and the correction
//   pressure row       and the stage-2 inverse, rebuilt from the product
//
// Needs a GPU and no MPI, HYPRE or CUTLASS.  Runs the FP32 product by default;
// with CUTLASS on a compute capability 8.0 device, pass tf32 to test the GEMM.
//
//   make TENSORCORE=1 test_tc_jacobian          # or build by hand, see tests/README.md
//   ./test_tc_jacobian [tf32]

#ifndef NVRS_HAVE_TENSORCORE
#define NVRS_HAVE_TENSORCORE 1
#endif
#include "../src/assemble_blackoil.cu"

#include <random>
#include <string>

using namespace nvrs;

// The host launcher in the included file names these; nothing here calls them.
namespace nvrs {
void halo_exchange(Real*, int, const Grid&) {}
void HaloConn::exchange(Real*, int) {}
}  // namespace nvrs
#ifdef NVRS_TEST_NO_MPI
extern "C" int MPI_Allreduce(const void*, void*, int, MPI_Datatype, MPI_Op,
                             MPI_Comm) { return 0; }
#endif

namespace {

int g_fail = 0;

void check(bool ok, const std::string& what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++g_fail;
}

template <class T>
T* dev(const std::vector<T>& h) {
  T* d = nullptr;
  NVRS_CHECK_CUDA(cudaMalloc(&d, sizeof(T) * std::max<size_t>(h.size(), 1)));
  if (!h.empty())
    NVRS_CHECK_CUDA(cudaMemcpy(d, h.data(), sizeof(T) * h.size(),
                               cudaMemcpyHostToDevice));
  return d;
}

template <class T>
std::vector<T> host(const T* d, size_t n) {
  std::vector<T> h(n);
  NVRS_CHECK_CUDA(cudaMemcpy(h.data(), d, sizeof(T) * n, cudaMemcpyDeviceToHost));
  return h;
}

struct Box {
  int nx = 5, ny = 4, nz = 3, cells = 60;
  std::vector<int> nbr, bcol, pcolpos, cellk, conn_ptr, conn_nbr, conn_bcol,
      conn_pcol, self_bcol, self_pcol, rowptr, prowptr;
  std::vector<Real> tface, conn_trans, conn_dz, conn_th;
  int nnz = 0, pnnz = 0;
};

// The same box in both descriptions.  Every row lists its self block first and
// then its neighbours, which is all the kernels ask of the column order.
Box make_box(int bs, std::mt19937& g) {
  Box B;
  std::uniform_real_distribution<double> U(0.2, 2.0);
  const int d[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
  B.nbr.assign(7 * B.cells, -1);
  B.bcol.assign(7 * B.cells, -1);
  B.pcolpos.assign(7 * B.cells, -1);
  B.tface.assign(6 * B.cells, 0);
  B.cellk.resize(B.cells);
  B.rowptr.assign(bs * B.cells + 1, 0);
  B.prowptr.assign(B.cells + 1, 0);
  B.conn_ptr.assign(B.cells + 1, 0);
  B.self_bcol.resize(B.cells);
  B.self_pcol.resize(B.cells);
  int nnz = 0, pnnz = 0;
  for (int k = 0, c = 0; k < B.nz; ++k)
    for (int j = 0; j < B.ny; ++j)
      for (int i = 0; i < B.nx; ++i, ++c) {
        B.cellk[c] = k;
        B.nbr[7 * c] = c;
        B.bcol[7 * c] = 0;
        B.pcolpos[7 * c] = 0;
        B.self_bcol[c] = 0;
        B.self_pcol[c] = 0;
        int deg = 1;
        for (int f = 0; f < 6; ++f) {
          const int I = i + d[f][0], J = j + d[f][1], K = k + d[f][2];
          if (I < 0 || J < 0 || K < 0 || I >= B.nx || J >= B.ny || K >= B.nz)
            continue;
          const int n = I + B.nx * (J + B.ny * K);
          const Real t = U(g);
          B.nbr[7 * c + 1 + f] = n;
          B.bcol[7 * c + 1 + f] = deg * bs;
          B.pcolpos[7 * c + 1 + f] = deg;
          B.tface[6 * c + f] = t;
          B.conn_nbr.push_back(n);
          B.conn_trans.push_back(t);
          B.conn_dz.push_back((Real)d[f][2] * 10);
          B.conn_th.push_back(U(g) < 0.5 ? (Real)U(g) : (Real)0);
          B.conn_bcol.push_back(deg * bs);
          B.conn_pcol.push_back(deg);
          ++deg;
        }
        B.conn_ptr[c + 1] = (int)B.conn_nbr.size();
        for (int a = 0; a < bs; ++a) {
          B.rowptr[bs * c + a] = nnz;
          nnz += deg * bs;
        }
        B.prowptr[c] = pnnz;
        pnnz += deg;
      }
  B.rowptr[bs * B.cells] = nnz;
  B.prowptr[B.cells] = pnnz;
  B.nnz = nnz;
  B.pnnz = pnnz;
  return B;
}

std::vector<Real> table_x(int n, Real lo, Real hi) {
  std::vector<Real> x(n);
  for (int i = 0; i < n; ++i) x[i] = lo + (hi - lo) * i / (n - 1);
  return x;
}

SatTables make_tables(bool pc) {
  const int n = 12;
  const auto sw = table_x(n, 0.1, 0.9), sg = table_x(n, 0.0, 0.8);
  std::vector<Real> krw(n), krow(n), krg(n), krog(n), pcow(n), pcog(n);
  for (int i = 0; i < n; ++i) {
    const Real a = (Real)i / (n - 1);
    krw[i] = a * a;
    krow[i] = (1 - a) * (1 - a);
    krg[i] = a * a * 0.9;
    krog[i] = (1 - a) * (1 - a);
    pcow[i] = pc ? 12 * (1 - a) : 0;
    pcog[i] = pc ? 6 * a : 0;
  }
  SatTables T{};
  T.sw = dev(sw); T.krw = dev(krw); T.krow = dev(krow); T.nsw = n;
  T.sg = dev(sg); T.krg = dev(krg); T.krog = dev(krog); T.nsg = n;
  T.pcow = dev(pcow); T.pcog = dev(pcog); T.have_pc = pc ? 1 : 0;
  set_table_endpoints(T, sw.data(), krw.data(), krow.data(), n, sg.data(),
                      krg.data(), krog.data(), n);
  return T;
}

struct Case {
  const char* name;
  int bs;
  bool conn;
  bool gas_primary;
};

void run_case(const Case& C, int precision, std::mt19937& g) {
  std::printf("\n%s\n", C.name);
  const int bs = C.bs;
  Box B = make_box(bs, g);
  const int cells = B.cells;
  std::uniform_real_distribution<double> U(0.0, 1.0);

  std::vector<Real> u(bs * cells), uo(bs * cells), poro(cells), aim(cells), vol(cells);
  std::vector<Real> prim(cells, 0), prim_old(cells, 0);
  for (int c = 0; c < cells; ++c) {
    u[bs * c] = 3000 + 300 * U(g);
    uo[bs * c] = u[bs * c] - 20 * U(g);
    u[bs * c + 1] = 0.2 + 0.4 * U(g);
    uo[bs * c + 1] = 0.2 + 0.4 * U(g);
    if (bs == 3) {
      u[bs * c + 2] = 0.05 + 0.2 * U(g);
      uo[bs * c + 2] = 0.05 + 0.2 * U(g);
    }
    poro[c] = 0.1 + 0.2 * U(g);
    vol[c] = 1000 * (1 + U(g));
    aim[c] = U(g) < 0.25 ? 0 : 1;
    if (C.gas_primary) {
      prim[c] = (Real)std::floor(3 * U(g));
      prim_old[c] = (Real)std::floor(3 * U(g));
    }
  }

  // Random PVT, so a term dropped from a lane cannot hide behind a zero slope.
  std::vector<AssemblyCellProps> props(cells);
  const auto s = [&](double scale) { return (Real)(scale * (2 * U(g) - 1)); };
  for (auto& q : props) {
    PvtCell& F = q.pvt;
    F.bo = 0.8 + 0.2 * U(g); F.bw = 0.95 + 0.05 * U(g); F.bg = 200 + 100 * U(g);
    F.dbo = s(1e-4); F.dbw = s(1e-5); F.dbg = s(1e-1);
    F.rs = 300 + 300 * U(g); F.drs = s(0.2);
    F.rv = 1e-4 * U(g); F.drv = s(1e-7);
    F.drv_dso = s(1e-4); F.dbg_dso = s(1.0); F.dmug_dso = s(1e-3);
    F.muo = 1 + U(g); F.muw = 0.4 + 0.2 * U(g); F.mug = 0.02 + 0.01 * U(g);
    F.dbo_dx = s(1e-2); F.dbg_dx = s(1.0); F.drs_dx = s(1.0); F.drv_dx = s(1e-5);
    F.dmuo_dx = s(1e-2); F.dmug_dx = s(1e-3);
    F.dbo_dsw = s(1e-2); F.dbg_dsw = s(1.0); F.drs_dsw = s(1.0); F.drv_dsw = s(1e-5);
    F.dmuo_dsw = s(1e-2); F.dmug_dsw = s(1e-3);
    q.cap.pcow = 5 * U(g); q.cap.dpcow = s(20);
    q.cap.pcog = 3 * U(g); q.cap.dpcog = s(10);
    q.mob.Mw = U(g); q.mob.Mo = U(g); q.mob.Mg = U(g);
    q.mob.dMw_dSw = s(2); q.mob.dMo_dSw = s(2); q.mob.dMo_dSg = s(2);
    q.mob.dMg_dSw = s(2); q.mob.dMg_dSg = s(2);
  }

  BOParams P{};
  P.bs = bs; P.nx = B.nx; P.ny = B.ny; P.nz = B.nz;
  P.dz_cell = 10; P.dt = 1; P.ct = 1e-5; P.uw = 0.5; P.uo = 1.5; P.ug = 0.03;
  P.bw = 1.02; P.bo = 1.2; P.bg = 0.005; P.rs = 0.3; P.swi = 0.1; P.swr = 0.1;
  P.rhow = 1.0; P.rhoo = 0.8; P.rhog = 0.1; P.gravity = 1; P.grav_grad = 0.433;
  P.fd_eps = 1e-6;

  Real* d_prim = dev(prim);
  Real* d_prim_old = dev(prim_old);
  if (C.gas_primary) { P.gas_primary = d_prim; P.gas_primary_old = d_prim_old; }

  const SatTables T = make_tables(true);
  Real* d_u = dev(u); Real* d_uo = dev(uo); Real* d_poro = dev(poro);
  Real* d_aim = dev(aim); Real* d_vol = dev(vol);
  int* d_nbr = dev(B.nbr); int* d_bcol = dev(B.bcol); int* d_pcp = dev(B.pcolpos);
  int* d_k = dev(B.cellk); int* d_cptr = dev(B.conn_ptr); int* d_cnbr = dev(B.conn_nbr);
  int* d_cbcol = dev(B.conn_bcol); int* d_cpcol = dev(B.conn_pcol);
  int* d_sb = dev(B.self_bcol); int* d_sp = dev(B.self_pcol);
  int* d_row = dev(B.rowptr); int* d_prow = dev(B.prowptr);
  Real* d_dz = dev(B.conn_dz); Real* d_th = dev(B.conn_th);
  const std::vector<WellCell> wells(cells);
  WellCell* d_well = dev(wells);
  AssemblyCellProps* d_props = dev(props);

  Real* d_b = dev(std::vector<Real>(bs * cells, 0));
  Real* d_a = dev(std::vector<Real>(B.nnz, 0));
  PrecReal* d_pv = dev(std::vector<PrecReal>(B.pnnz, 0));
  PrecReal* d_di = dev(std::vector<PrecReal>(bs * bs * cells, 0));
  PrecReal* d_pw = dev(std::vector<PrecReal>(bs * cells, 0));
  Real* d_rn = dev(std::vector<Real>(23, 0));
  unsigned long long* d_cnv = dev(std::vector<unsigned long long>(3, 0));

  const auto analytic = [&](const Real* tface, const Real* trans) {
    const Real V = 1000;
    if (bs == 2 && C.conn)
      k_assemble_bo2_analytic_conn<<<grid1d(cells), 256>>>(
          d_u, d_uo, d_poro, d_cptr, d_cnbr, trans, d_dz, d_th, d_cbcol, d_cpcol,
          d_sb, d_sp, d_row, d_prow, d_well, d_aim, d_vol, T, P, V, d_b, d_a, d_pv,
          d_di, d_pw, d_rn, d_cnv);
    else if (bs == 2)
      k_assemble_bo_analytic<<<grid1d(cells), 256>>>(
          d_u, d_uo, d_poro, tface, d_nbr, d_k, d_row, d_bcol, d_prow, d_pcp, d_well,
          d_aim, T, P, V, d_b, d_a, d_pv, d_di, d_pw, d_rn);
    else if (C.conn)
      k_assemble_bo3_analytic_conn<<<grid1d(cells), 256>>>(
          d_u, d_uo, d_poro, d_cptr, d_cnbr, trans, d_dz, d_th, d_cbcol, d_cpcol,
          d_sb, d_sp, d_row, d_prow, d_well, d_aim, d_vol, d_props, T, P, V, d_b,
          d_a, d_pv, d_di, d_pw, d_rn, d_cnv);
    else
      k_assemble_bo3_analytic<<<grid1d(cells), 256>>>(
          d_u, d_uo, d_poro, tface, d_nbr, d_k, d_row, d_bcol, d_prow, d_pcp, d_well,
          d_aim, T, P, V, d_b, d_a, d_pv, d_di, d_pw, d_rn);
    NVRS_CHECK_CUDA(cudaDeviceSynchronize());
  };

  // Accumulation alone, then the full analytic system.
  Real* d_t0 = dev(std::vector<Real>(B.tface.size(), 0));
  Real* d_c0 = dev(std::vector<Real>(B.conn_trans.size(), 0));
  analytic(d_t0, d_c0);
  const auto a_base = host(d_a, B.nnz);
  Real* d_tf = dev(B.tface);
  Real* d_ct = dev(B.conn_trans);
  analytic(d_tf, d_ct);
  const auto a_full = host(d_a, B.nnz);
  const auto pv_full = host(d_pv, B.pnnz);
  const auto di_full = host(d_di, bs * bs * cells);
  std::vector<int> owner(B.conn_nbr.size());
  for (int c = 0; c < cells; ++c)
    for (int m = B.conn_ptr[c]; m < B.conn_ptr[c + 1]; ++m) owner[m] = c;

  // The TC state, with a small batch so that several batches are exercised.
  g_tc.conn_owner = dev(owner);
  g_tc.nconn = C.conn ? (long long)B.conn_nbr.size() : 6LL * cells;
  g_tc.capacity = 37;
  const size_t tile_b = (size_t)g_tc.capacity * tc::kTile * tc::kTile * sizeof(float);
  NVRS_CHECK_CUDA(cudaMalloc(&g_tc.lflux, tile_b));
  NVRS_CHECK_CUDA(cudaMalloc(&g_tc.lprops, tile_b));
  NVRS_CHECK_CUDA(cudaMalloc(&g_tc.prod, tile_b));
  NVRS_CHECK_CUDA(cudaMalloc(&g_tc.dself, sizeof(Real) * (size_t)cells * bs * bs));
  NVRS_CHECK_CUDA(cudaMalloc(&g_tc.stats, sizeof(Real) * 4));
  NVRS_CHECK_CUDA(cudaMalloc(&g_tc.worst, sizeof(unsigned long long) * 2));
  g_tc.v_neighbour = g_tc.v_self = g_tc.v_worst_neighbour = g_tc.v_worst_self = 0;

  TcView V{};
  V.u = d_u; V.u_old = d_uo; V.aimflag = d_aim; V.conn = C.conn;
  V.conn_ptr = d_cptr; V.conn_owner = g_tc.conn_owner; V.conn_nbr = d_cnbr;
  V.conn_trans = d_ct; V.conn_dz = d_dz; V.conn_threshold = d_th;
  V.conn_bcol = d_cbcol; V.conn_pcol = d_cpcol; V.self_bcol = d_sb; V.self_pcol = d_sp;
  V.tface = d_tf; V.nbr = d_nbr; V.cellk = d_k; V.bcol = d_bcol; V.pcolpos = d_pcp;
  V.rowptr = d_row; V.prowptr = d_prow; V.props = d_props; V.T = T; V.P = P;
  V.lflux = g_tc.lflux; V.lprops = g_tc.lprops; V.prod = g_tc.prod;
  V.dself = g_tc.dself; V.aval = d_a; V.pval = d_pv; V.dinv = d_di; V.pweight = d_pw;
  V.stats = g_tc.stats; V.worst = g_tc.worst;

  // 1. The factorisation itself, in FP64: accumulation + sum of L_flux L_props
  //    must equal the full analytic self block.  Fill only, no product.
  NVRS_CHECK_CUDA(cudaMemset(g_tc.dself, 0, sizeof(Real) * (size_t)cells * bs * bs));
  for (long long first = 0; first < g_tc.nconn; first += g_tc.capacity) {
    const int count = (int)std::min<long long>(g_tc.capacity, g_tc.nconn - first);
    const dim3 grid = grid1d(count);
    if (bs == 2 && C.conn)  k_tc_fill_bo2_conn<<<grid, 256>>>(V, first, count);
    else if (bs == 2)       k_tc_fill_bo2_cart<<<grid, 256>>>(V, first, count);
    else if (C.conn)        k_tc_fill_bo3_conn<<<grid, 256>>>(V, first, count);
    else                    k_tc_fill_bo3_cart<<<grid, 256>>>(V, first, count);
    NVRS_CHECK_CUDA(cudaDeviceSynchronize());
  }
  const auto dself = host(g_tc.dself, (size_t)cells * bs * bs);
  double worst_fp64 = 0, norm = 0;
  for (int c = 0; c < cells; ++c)
    for (int a = 0; a < bs; ++a)
      for (int v = 0; v < bs; ++v) {
        const int e = B.rowptr[bs * c + a] + v;
        const double conn_part = a_full[e] - a_base[e];
        worst_fp64 = std::max(worst_fp64, std::fabs(conn_part + dself[bs * bs * c + a * bs + v]));
        norm = std::max(norm, std::fabs(conn_part));
      }
  char buf[160];
  std::snprintf(buf, sizeof buf, "self block, FP64 factorisation: worst %.2e of %.2e",
                worst_fp64, norm);
  check(worst_fp64 <= 1e-10 * norm, buf);

  // 2. The whole path: product, validate, scatter, finish.
  tc::g_validate = 1;
  tc_run_connections(V, cells, bs);
  const auto a_tc = host(d_a, B.nnz);
  const auto pv_tc = host(d_pv, B.pnnz);
  const auto di_tc = host(d_di, bs * bs * cells);

  // Entry by entry, relative to the largest entry of the row.  TF32 rounds its
  // inputs to 11 significant bits, FP32 to 24.
  const double tol = precision == tc::kPrecTF32 ? 5e-2 : 2e-4;
  double worst_nb = 0, worst_self = 0;
  for (int c = 0; c < cells; ++c) {
    const int deg = (B.rowptr[bs * c + 1] - B.rowptr[bs * c]) / bs;
    for (int a = 0; a < bs; ++a) {
      const int r = B.rowptr[bs * c + a];
      double rowmax = 0;
      for (int e = 0; e < deg * bs; ++e) rowmax = std::max(rowmax, std::fabs(a_full[r + e]));
      if (rowmax == 0) continue;
      for (int e = 0; e < deg * bs; ++e) {
        const double d = std::fabs(a_tc[r + e] - a_full[r + e]) / rowmax;
        (e < bs ? worst_self : worst_nb) = std::max(e < bs ? worst_self : worst_nb, d);
      }
    }
  }
  std::snprintf(buf, sizeof buf, "neighbour blocks through the product: worst %.2e", worst_nb);
  check(worst_nb <= tol, buf);
  std::snprintf(buf, sizeof buf, "self blocks through the product:      worst %.2e", worst_self);
  check(worst_self <= tol, buf);

  double worst_pv = 0, pvmax = 0, worst_di = 0, dimax = 0;
  for (int e = 0; e < B.pnnz; ++e) {
    worst_pv = std::max(worst_pv, (double)std::fabs(pv_tc[e] - pv_full[e]));
    pvmax = std::max(pvmax, (double)std::fabs(pv_full[e]));
  }
  for (size_t e = 0; e < di_full.size(); ++e) {
    worst_di = std::max(worst_di, (double)std::fabs(di_tc[e] - di_full[e]));
    dimax = std::max(dimax, (double)std::fabs(di_full[e]));
  }
  std::snprintf(buf, sizeof buf, "pressure row %.2e, stage-2 inverse %.2e (relative)",
                worst_pv / pvmax, worst_di / dimax);
  check(worst_pv <= 2e3 * tol * pvmax && worst_di <= 2e3 * tol * dimax, buf);
  std::snprintf(buf, sizeof buf, "TCVALIDATE reports neighbour %.2e, self %.2e",
                g_tc.v_neighbour, g_tc.v_self);
  check(g_tc.v_neighbour > 0 && g_tc.v_neighbour <= tol && g_tc.v_self <= tol, buf);

  // A row of zeros in the tile beyond the physical block must stay zero.
  const auto tile = host(g_tc.prod, (size_t)tc::kTile * tc::kTile);
  double pad = 0;
  for (int r = bs; r < tc::kTile; ++r)
    for (int c = 0; c < tc::kTile; ++c) pad = std::max(pad, (double)std::fabs(tile[r * tc::kTile + c]));
  check(pad == 0, "padding lanes of the product are zero");

  cudaFree(g_tc.lflux); cudaFree(g_tc.lprops); cudaFree(g_tc.prod);
  cudaFree(g_tc.dself); cudaFree(g_tc.stats); cudaFree(g_tc.worst);
  cudaFree(g_tc.conn_owner);
  g_tc.lflux = nullptr;
}

}  // namespace

int main(int argc, char** argv) {
  const int precision = (argc > 1 && std::string(argv[1]) == "tf32")
                            ? tc::kPrecTF32 : tc::kPrecFP32;
  tc::enable(precision, 1);
  std::printf("product: %s\n", tc::backend_name(precision));
  std::mt19937 g(20261030);
  const Case cases[] = {
      {"two-phase, connection list", 2, true, false},
      {"two-phase, fixed stencil", 2, false, false},
      {"three-phase, connection list, Sg primary", 3, true, false},
      {"three-phase, connection list, Sg/Rs/Rv switching", 3, true, true},
      {"three-phase, fixed stencil", 3, false, false},
  };
  for (const Case& c : cases) run_case(c, precision, g);
  std::printf("\n%s\n", g_fail ? "FAILED" : "all passed");
  return g_fail ? 1 : 0;
}
