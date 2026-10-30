"""
mixed_precision_cpr_amgx.py
===========================
Reference implementation of mixed-precision CPR for compositional reservoir
simulation using CuPy (state / outer Krylov, FP64) and AmgX (CPR / AMG
preconditioner, FP32 hierarchy), with zero-copy device-pointer interop.

Realizes the precision map from the paper:
    * FP64 : state, true residual b - J dx, Krylov vectors, dot products (the u_r / u slots)
    * FP32 : preconditioner matrix (off-diagonals), AMG hierarchy, smoother (the u_f slot)

Two entry points:
    (A) solve_amgx_mixed(...)  -- let AmgX do FGMRES+CPR in a mixed mode (dDFI:
                                  FP64 vectors, FP32 matrix). One call, minimal code.
    (B) fgmres_ir(...)         -- hand-rolled FP64 flexible GMRES whose preconditioner
                                  application is a single AmgX float CPR/AMG cycle.
                                  Keeps the FP64 master J for the true residual and the
                                  lumped FP32 A_hat only inside M^{-1}.

Tested for shape/dtype correctness; requires a working AmgX shared library
(libamgxsh.so) and a CUDA GPU at run time. Index type is 32-bit (the 'I' suffix).

Authors: C. Etienam, N. Luiken, I. Said.
"""

import ctypes
import ctypes.util
import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
except Exception:  # allow import on a CPU box for inspection
    cp = None
    cpsp = None


# ---------------------------------------------------------------------------
# 1. Minimal ctypes binding to libamgxsh.so
# ---------------------------------------------------------------------------
def _load_amgx():
    name = ctypes.util.find_library("amgxsh") or "libamgxsh.so"
    return ctypes.cdll.LoadLibrary(name)


_AMGX = None  # lazy global handle


def _amgx():
    global _AMGX
    if _AMGX is None:
        _AMGX = _load_amgx()
        _decl(_AMGX)
    return _AMGX


def _decl(lib):
    """Set argtypes for the subset of the AmgX C API we use."""
    P = ctypes.c_void_p
    PP = ctypes.POINTER(ctypes.c_void_p)
    I = ctypes.c_int
    cs = ctypes.c_char_p
    lib.AMGX_initialize.argtypes = []
    lib.AMGX_finalize.argtypes = []
    lib.AMGX_config_create.argtypes = [PP, cs]
    lib.AMGX_config_destroy.argtypes = [P]
    lib.AMGX_resources_create_simple.argtypes = [PP, P]
    lib.AMGX_resources_destroy.argtypes = [P]
    lib.AMGX_matrix_create.argtypes = [PP, P, I]
    lib.AMGX_matrix_destroy.argtypes = [P]
    lib.AMGX_vector_create.argtypes = [PP, P, I]
    lib.AMGX_vector_destroy.argtypes = [P]
    lib.AMGX_solver_create.argtypes = [PP, P, I, P]
    lib.AMGX_solver_destroy.argtypes = [P]
    # upload_all(mtx, n, nnz, bx, by, row_ptr, col_idx, data, diag_data)
    lib.AMGX_matrix_upload_all.argtypes = [P, I, I, I, I, P, P, P, P]
    # replace_coefficients(mtx, n, nnz, data, diag_data)  -- reuse structure across Newton steps
    lib.AMGX_matrix_replace_coefficients.argtypes = [P, I, I, P, P]
    lib.AMGX_vector_upload.argtypes = [P, I, I, P]
    lib.AMGX_vector_download.argtypes = [P, P]
    lib.AMGX_vector_set_zero.argtypes = [P, I, I]
    lib.AMGX_solver_setup.argtypes = [P, P]
    lib.AMGX_solver_solve.argtypes = [P, P, P]
    lib.AMGX_solver_get_iterations_number.argtypes = [P, ctypes.POINTER(I)]


def _check(rc, where):
    if rc != 0:
        raise RuntimeError(f"AmgX call failed (rc={rc}) at {where}")


# AmgX mode enum values. These are the AMGX_mode_* integers, which are
# build-specific; the ones below were read from the AMGX 2.5.0 amgx_c.h built on
# this system (compile a 6-line C program printing AMGX_mode_dDDI etc. to confirm
# for a different build). Naming is [memory][VECprec][MATprec][index]:
#   dDDI : device, FP64 vector, FP64 matrix, int32 index   (all double)
#   dDFI : device, FP64 vector, FP32 matrix, int32 index   (MIXED -- our path)
#   dFFI : device, FP32 vector, FP32 matrix, int32 index   (all float)
AMGX_MODE = {"dDDI": 8193, "dDFI": 8449, "dFFI": 8465}


class AmgX:
    """RAII-ish wrapper around one AmgX resources/matrix/vector/solver set."""

    def __init__(self, config_json: str, mode: str = "dDFI"):
        self.lib = _amgx()
        self.mode = AMGX_MODE[mode]
        self._initialized = False
        self.cfg = ctypes.c_void_p()
        self.rsc = ctypes.c_void_p()
        self.A = ctypes.c_void_p()
        self.b = ctypes.c_void_p()
        self.x = ctypes.c_void_p()
        self.slv = ctypes.c_void_p()
        self._n = 0
        self._bs = 1

        _check(self.lib.AMGX_initialize(), "initialize")
        self._initialized = True
        _check(self.lib.AMGX_config_create(ctypes.byref(self.cfg),
                                           config_json.encode()), "config_create")
        _check(self.lib.AMGX_resources_create_simple(ctypes.byref(self.rsc), self.cfg),
               "resources_create")
        for h in (self.A,):
            _check(self.lib.AMGX_matrix_create(ctypes.byref(h), self.rsc, self.mode),
                   "matrix_create")
        for h in (self.b, self.x):
            _check(self.lib.AMGX_vector_create(ctypes.byref(h), self.rsc, self.mode),
                   "vector_create")
        _check(self.lib.AMGX_solver_create(ctypes.byref(self.slv), self.rsc,
                                           self.mode, self.cfg), "solver_create")

    # ---- matrix value precision implied by mode: dDFI/dFFI -> float32 ----
    @property
    def _mat_dtype(self):
        return np.float64 if self.mode == AMGX_MODE["dDDI"] else np.float32

    @property
    def _vec_dtype(self):
        return np.float32 if self.mode == AMGX_MODE["dFFI"] else np.float64

    def upload(self, Acsr, block_size: int = 1):
        """Upload a (CuPy) CSR matrix by DEVICE pointer (zero host round-trip).

        Acsr.data must already be in self._mat_dtype; indices/indptr int32.
        For block systems pass a block-CSR with block_size = n_u.
        """
        self._n = (Acsr.shape[0] // block_size)
        self._bs = block_size
        nnz_b = Acsr.nnz // (block_size * block_size) if block_size > 1 else Acsr.nnz
        rp = _ptr(Acsr.indptr)        # int32 device ptr
        ci = _ptr(Acsr.indices)       # int32 device ptr
        va = _ptr(Acsr.data)          # value device ptr (float32 for mixed)
        _check(self.lib.AMGX_matrix_upload_all(
            self.A, self._n, nnz_b, block_size, block_size, rp, ci, va, None),
            "matrix_upload_all")
        _check(self.lib.AMGX_solver_setup(self.slv, self.A), "solver_setup")

    def replace(self, Acsr):
        """Reuse the hierarchy structure across Newton steps; only values change."""
        nnz_b = Acsr.nnz // (self._bs * self._bs) if self._bs > 1 else Acsr.nnz
        _check(self.lib.AMGX_matrix_replace_coefficients(
            self.A, self._n, nnz_b, _ptr(Acsr.data), None), "replace_coefficients")
        _check(self.lib.AMGX_solver_setup(self.slv, self.A), "resetup")

    def solve(self, b_dev, x_dev):
        """Solve / apply: x_dev <- (solver) applied to b_dev. Zero-copy device ptrs.

        With a max_iters=1 single-cycle config this is one application of M^{-1}
        (the path-B preconditioner). With a full FGMRES+CPR config it is a full solve
        (path A). b_dev, x_dev are CuPy arrays of self._vec_dtype.
        """
        n, bs = self._n, self._bs
        _check(self.lib.AMGX_vector_upload(self.b, n, bs, _ptr(b_dev)), "vec_upload b")
        _check(self.lib.AMGX_vector_upload(self.x, n, bs, _ptr(x_dev)), "vec_upload x")
        _check(self.lib.AMGX_solver_solve(self.slv, self.b, self.x), "solve")
        _check(self.lib.AMGX_vector_download(self.x, _ptr(x_dev)), "vec_download")
        it = ctypes.c_int(0)
        self.lib.AMGX_solver_get_iterations_number(self.slv, ctypes.byref(it))
        return it.value

    def destroy(self):
        L = self.lib
        for fn, h in ((L.AMGX_solver_destroy, self.slv), (L.AMGX_vector_destroy, self.x),
                      (L.AMGX_vector_destroy, self.b), (L.AMGX_matrix_destroy, self.A),
                      (L.AMGX_resources_destroy, self.rsc),
                      (L.AMGX_config_destroy, self.cfg)):
            if h:
                fn(h)
        if self._initialized:
            L.AMGX_finalize()
            self._initialized = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.destroy()


def _ptr(cupy_arr):
    """CuPy array -> ctypes c_void_p device pointer (zero-copy).

    AmgX copies device->device via cudaMemcpyDefault, so a device pointer is
    accepted directly -- no host staging buffer."""
    return ctypes.c_void_p(int(cupy_arr.data.ptr))


# ---------------------------------------------------------------------------
# 2. Diagonal-lumped narrowing cast  (Eq. 6 of the paper), scalar CSR
#    off-diagonals -> FP32; rounding error lumped onto the (FP32-stored)
#    diagonal so that row sums are preserved => AMG keeps the constant null space.
# ---------------------------------------------------------------------------
def cast_lump_scalar(J64):
    """J64: CuPy FP64 csr_matrix -> FP32 csr_matrix A_hat with lumped diagonal."""
    n = J64.shape[0]
    d = J64.diagonal().astype(cp.float64)                 # FP64 diagonal
    Joff = J64 - cpsp.diags(d)                            # off-diagonal part (FP64)
    Joff.eliminate_zeros()
    a32 = Joff.data.astype(cp.float32)                    # narrow off-diagonals
    dropped = Joff.data - a32.astype(cp.float64)          # rounding residual (FP64)
    row_ids = cp.repeat(cp.arange(n), cp.diff(Joff.indptr))
    rowsum = cp.zeros(n, dtype=cp.float64)
    cp.add.at(rowsum, row_ids, dropped)                   # per-row lumped mass (FP64)
    d_corr = (d + rowsum).astype(cp.float32)              # corrected diagonal -> FP32
    Ahat = Joff.astype(cp.float32)
    Ahat = Ahat + cpsp.diags(d_corr).astype(cp.float32)   # reinsert FP32 diagonal
    Ahat = cpsp.csr_matrix(Ahat)
    Ahat.indptr = Ahat.indptr.astype(cp.int32)
    Ahat.indices = Ahat.indices.astype(cp.int32)
    return Ahat


# ---------------------------------------------------------------------------
# 3. Path A: single-call mixed-precision AmgX solve (FP64 vectors, FP32 matrix)
# ---------------------------------------------------------------------------
def solve_amgx_mixed(J64, b64, config_json, block_size=1, tol=1e-8):
    """Upload FP32 (lumped) matrix + FP64 RHS in dDFI; AmgX runs FGMRES+CPR.
    Returns (x_fp64, iters). The hierarchy is FP32; vectors stay FP64 -> the
    outer accuracy is FP64 while the heavy SpMV/AMG traffic is FP32."""
    Ahat = cast_lump_scalar(J64) if block_size == 1 else J64.astype(cp.float32)
    with AmgX(config_json, mode="dDFI") as solver:
        solver.upload(Ahat, block_size=block_size)
        x = cp.zeros_like(b64, dtype=cp.float64)
        b = cp.ascontiguousarray(b64, dtype=cp.float64)
        iters = solver.solve(b, x)
    return x, iters


# ---------------------------------------------------------------------------
# 4. Path B: FP64 flexible GMRES with AmgX float CPR/AMG as preconditioner
#    M^{-1} r is ONE AmgX cycle (config with max_iters=1). True residual uses
#    the FP64 master J; only the preconditioner sees FP32.
# ---------------------------------------------------------------------------
def fgmres_ir(J64, b64, precond, tol=1e-8, restart=50, maxit=500):
    """J64: FP64 CuPy csr_matrix (master).  precond: callable r64 -> z64 (AmgX dDFI).
    Returns (x, residual_history). All Krylov arithmetic FP64; dots FP64-accumulated."""
    n = b64.size
    x = cp.zeros(n, dtype=cp.float64)
    r = b64 - J64 @ x                                     # true residual, FP64 (u_r slot)
    bnorm = float(cp.linalg.norm(b64))
    res_hist = []
    for _outer in range(maxit // restart + 1):
        beta = float(cp.linalg.norm(r))
        res_hist.append(beta / max(bnorm, 1e-300))
        if beta <= tol * bnorm:
            break
        V = cp.zeros((restart + 1, n), dtype=cp.float64)  # Arnoldi basis (FP64)
        Z = cp.zeros((restart, n), dtype=cp.float64)      # FLEXIBLE: store M^{-1} v_j
        H = cp.zeros((restart + 1, restart), dtype=cp.float64)
        g = cp.zeros(restart + 1, dtype=cp.float64)
        g[0] = beta
        V[0] = r / beta
        cs = cp.zeros(restart); sn = cp.zeros(restart)
        j_used = 0
        for j in range(restart):
            Z[j] = precond(V[j])                          # u_f slot: AmgX FP32 cycle
            w = J64 @ Z[j]                                # FP64 master SpMV
            for i in range(j + 1):
                H[i, j] = float(cp.dot(w, V[i]))          # FP64-accumulated dot
                w = w - H[i, j] * V[i]
            H[j + 1, j] = float(cp.linalg.norm(w))
            if H[j + 1, j] > 1e-300:
                V[j + 1] = w / H[j + 1, j]
            # apply previous Givens rotations to new column
            for i in range(j):
                t = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = t
            denom = float(cp.hypot(H[j, j], H[j + 1, j]))
            cs[j] = H[j, j] / denom; sn[j] = H[j + 1, j] / denom
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0
            g[j + 1] = -sn[j] * g[j]; g[j] = cs[j] * g[j]
            j_used = j + 1
            if abs(float(g[j + 1])) <= tol * bnorm:
                break
        # solve upper-triangular H y = g  and update x with the FLEXIBLE basis Z
        y = cp.zeros(j_used, dtype=cp.float64)
        for i in range(j_used - 1, -1, -1):
            s = g[i] - cp.dot(H[i, i + 1:j_used], y[i + 1:j_used])
            y[i] = s / H[i, i]
        x = x + Z[:j_used].T @ y
        r = b64 - J64 @ x                                 # recompute true residual FP64
    res_hist.append(float(cp.linalg.norm(r)) / max(bnorm, 1e-300))
    return x, res_hist


def make_amgx_preconditioner(J64, config_json_one_cycle, block_size=1):
    """Build an AmgX all-FP32 (dFFI) single-cycle AMG and return a callable r64->z64.

    AmgX dropped the mixed dDFI mode on CUDA>=10.1, so the preconditioner runs in
    pure FP32 (dFFI): the FP64 residual is cast to FP32 in, one AMG cycle runs in
    FP32, and the FP32 result is cast back to FP64. The OUTER FGMRES (fgmres_ir)
    stays FP64 with the FP64 master matrix, so the final solution is FP64-accurate.
    """
    Ahat = cast_lump_scalar(J64) if block_size == 1 else J64.astype(cp.float32)
    solver = AmgX(config_json_one_cycle, mode="dFFI")     # all-float (supported)
    solver.upload(Ahat, block_size=block_size)
    n = J64.shape[0]
    r32 = cp.zeros(n, dtype=cp.float32)
    z32 = cp.zeros(n, dtype=cp.float32)

    def apply(r64):
        r32[...] = r64.astype(cp.float32)                 # FP64 residual -> FP32
        z32.fill(0.0)
        solver.solve(r32, z32)                            # one FP32 M^{-1} cycle
        return z32.astype(cp.float64)                     # FP32 -> FP64 for outer loop

    apply.solver = solver  # keep handle alive; call apply.solver.destroy() when done
    return apply


# ---------------------------------------------------------------------------
# 5. BONUS: hand-rolled block-CSR mixed-precision SpMV (FP64 diag, FP32 off-diag,
#    FP64 accumulation) for a custom CPR that does not route through AmgX.
#    Diagonal blocks stored separately (one per block-row, FP64); off-diagonals
#    in a float32 block-CSR. "store narrow, accumulate wide."
# ---------------------------------------------------------------------------
_BSR_MIXED_SRC = r"""
extern "C" __global__
void bsr_spmv_mixed(const int n_brow, const int nu,
                    const int*    __restrict__ off_rowptr,
                    const int*    __restrict__ off_colind,
                    const float*  __restrict__ off_val,    // nnz_off * nu*nu, FP32
                    const double* __restrict__ diag_val,   // n_brow * nu*nu, FP64
                    const double* __restrict__ x,          // n_brow * nu,     FP64
                    double*       __restrict__ y)          // n_brow * nu,     FP64
{
    int bi = blockIdx.x * blockDim.x + threadIdx.x;        // one thread per block-row
    if (bi >= n_brow) return;
    double acc[32];                                        // nu <= 32 assumed
    for (int r = 0; r < nu; ++r) acc[r] = 0.0;
    // diagonal block: FP64 x FP64
    const double* D = diag_val + (size_t)bi * nu * nu;
    const double* xb = x + (size_t)bi * nu;
    for (int r = 0; r < nu; ++r)
        for (int c = 0; c < nu; ++c)
            acc[r] += D[r*nu + c] * xb[c];
    // off-diagonal blocks: FP32 read, FP64 multiply-add
    for (int p = off_rowptr[bi]; p < off_rowptr[bi+1]; ++p) {
        int bj = off_colind[p];
        const float*  B  = off_val + (size_t)p * nu * nu;
        const double* xj = x + (size_t)bj * nu;
        for (int r = 0; r < nu; ++r)
            for (int c = 0; c < nu; ++c)
                acc[r] += (double)B[r*nu + c] * xj[c];     // wide accumulate
    }
    double* yb = y + (size_t)bi * nu;
    for (int r = 0; r < nu; ++r) yb[r] = acc[r];
}
"""


def bsr_spmv_mixed(off_rowptr, off_colind, off_val_f32, diag_val_f64, x_f64, nu):
    """y = A x with FP64 diagonal blocks, FP32 off-diagonal blocks, FP64 accumulation."""
    n_brow = off_rowptr.size - 1
    y = cp.empty(n_brow * nu, dtype=cp.float64)
    ker = cp.RawKernel(_BSR_MIXED_SRC, "bsr_spmv_mixed")
    threads = 128
    blocks = (n_brow + threads - 1) // threads
    ker((blocks,), (threads,),
        (np.int32(n_brow), np.int32(nu),
         off_rowptr.astype(cp.int32), off_colind.astype(cp.int32),
         off_val_f32.astype(cp.float32), diag_val_f64.astype(cp.float64),
         x_f64.astype(cp.float64), y))
    return y


# ---------------------------------------------------------------------------
# 6. AmgX configs.  CPR-style FGMRES (verify CPR keys against your AmgX build);
#    a robust AMG-preconditioned FGMRES is given as a guaranteed-valid fallback.
# ---------------------------------------------------------------------------
# Full solve (path A): FGMRES + classical-AMG preconditioner on the pressure-like system.
CONFIG_FGMRES_AMG = r"""
{
  "config_version": 2,
  "determinism_flag": 1,
  "solver": {
    "solver": "FGMRES", "gmres_n_restart": 50, "max_iters": 200,
    "tolerance": 1e-8, "convergence": "RELATIVE_INI_CORE",
    "monitor_residual": 1, "norm": "L2",
    "preconditioner": {
      "solver": "AMG", "algorithm": "CLASSICAL", "max_iters": 1,
      "cycle": "V", "presweeps": 1, "postsweeps": 1,
      "smoother": { "solver": "BLOCK_JACOBI", "relaxation_factor": 0.8 },
      "coarse_solver": "DENSE_LU_SOLVER",
      "max_levels": 24, "interpolator": "D2", "selector": "PMIS"
    }
  }
}
"""

# Two-stage CPR for compositional block systems.  AmgX exposes CPR via a nested
# preconditioner; the exact "CPR_*" keys depend on your AmgX version -- confirm.
CONFIG_FGMRES_CPR = r"""
{
  "config_version": 2,
  "solver": {
    "solver": "FGMRES", "gmres_n_restart": 50, "max_iters": 200,
    "tolerance": 1e-8, "convergence": "RELATIVE_INI_CORE", "monitor_residual": 1,
    "preconditioner": {
      "solver": "CPR", "max_iters": 1,
      "pressure_solver": {
        "solver": "AMG", "algorithm": "AGGREGATION", "max_iters": 1,
        "cycle": "V", "presweeps": 1, "postsweeps": 1,
        "smoother": "BLOCK_JACOBI", "coarse_solver": "DENSE_LU_SOLVER"
      },
      "second_stage_smoother": { "solver": "BLOCK_ILU0", "max_iters": 1 }
    }
  }
}
"""

# Single-cycle preconditioner (path B): solver IS the AMG, max_iters=1, no inner Krylov,
# residual monitoring off so AMGX_solver_solve applies exactly one M^{-1} cycle.
CONFIG_ONECYCLE_AMG = r"""
{
  "config_version": 2,
  "solver": {
    "solver": "AMG", "algorithm": "CLASSICAL", "max_iters": 1,
    "cycle": "V", "presweeps": 1, "postsweeps": 1, "monitor_residual": 0,
    "smoother": { "solver": "BLOCK_JACOBI", "relaxation_factor": 0.8 },
    "coarse_solver": "DENSE_LU_SOLVER", "selector": "PMIS", "interpolator": "D2"
  }
}
"""

# ---- FP32-SAFE variants ----------------------------------------------------
# AmgX's FP32 DENSE_LU_SOLVER coarse solve crashes (cudense_getrf, illegal
# memory access) on this Blackwell/CUDA build. These configs avoid the dense-LU
# coarse solve: cap the hierarchy small and use a few BLOCK_JACOBI sweeps on the
# coarsest level instead (coarse_solver = NOSOLVER means "just smooth"). Use
# these for dFFI (all-float) AmgX so setup does not reach cudense_getrf.
CONFIG_FGMRES_AMG_F32SAFE = r"""
{
  "config_version": 2,
  "determinism_flag": 1,
  "solver": {
    "solver": "FGMRES", "gmres_n_restart": 50, "max_iters": 200,
    "tolerance": 1e-6, "convergence": "RELATIVE_INI_CORE",
    "monitor_residual": 1, "norm": "L2",
    "preconditioner": {
      "solver": "AMG", "algorithm": "CLASSICAL", "max_iters": 1,
      "cycle": "V", "presweeps": 2, "postsweeps": 2,
      "smoother": { "solver": "BLOCK_JACOBI", "relaxation_factor": 0.7 },
      "coarse_solver": "NOSOLVER",
      "min_coarse_rows": 32, "max_levels": 18,
      "interpolator": "D2", "selector": "PMIS"
    }
  }
}
"""

CONFIG_ONECYCLE_AMG_F32SAFE = r"""
{
  "config_version": 2,
  "solver": {
    "solver": "AMG", "algorithm": "CLASSICAL", "max_iters": 1,
    "cycle": "V", "presweeps": 2, "postsweeps": 2, "monitor_residual": 0,
    "smoother": { "solver": "BLOCK_JACOBI", "relaxation_factor": 0.7 },
    "coarse_solver": "NOSOLVER",
    "min_coarse_rows": 32, "max_levels": 18,
    "selector": "PMIS", "interpolator": "D2"
  }
}
"""


# ---------------------------------------------------------------------------
# 7. Demo  (runs only with CuPy + AmgX + a GPU present)
# ---------------------------------------------------------------------------
def _demo():
    if cp is None:
        print("CuPy not available -- import-only inspection mode.")
        return
    # Synthetic SPD-ish 3D 7-point Poisson-like pressure operator as a stand-in for A_p.
    nx = ny = nz = 24
    n = nx * ny * nz
    import cupyx.scipy.sparse as sp
    main = 6.0 * cp.ones(n)
    offs, diags = [main], [0]
    for stride in (1, nx, nx * ny):
        e = -1.0 * cp.ones(n)
        offs += [e, e]; diags += [stride, -stride]
    J64 = sp.diags(offs, diags, shape=(n, n), format="csr").astype(cp.float64)
    J64.indptr = J64.indptr.astype(cp.int32); J64.indices = J64.indices.astype(cp.int32)
    b64 = cp.ones(n, dtype=cp.float64)

    print(f"n = {n}, nnz = {J64.nnz}")
    print(f"cast_lump: row-sum max |drift| = "
          f"{float(cp.abs((J64 @ cp.ones(n)) - (cast_lump_scalar(J64).astype(cp.float64) @ cp.ones(n))).max()):.2e}")

    try:
        # Path A: one AmgX mixed-precision solve
        xA, itA = solve_amgx_mixed(J64, b64, CONFIG_FGMRES_AMG)
        print(f"[A] AmgX mixed FGMRES+AMG: {itA} iters, "
              f"||b-Jx||/||b|| = {float(cp.linalg.norm(b64 - J64 @ xA) / cp.linalg.norm(b64)):.2e}")

        # Path B: FP64 FGMRES-IR with AmgX float one-cycle preconditioner
        M = make_amgx_preconditioner(J64, CONFIG_ONECYCLE_AMG)
        xB, hist = fgmres_ir(J64, b64, M, tol=1e-8, restart=50, maxit=500)
        M.solver.destroy()
        print(f"[B] FP64 FGMRES-IR (AmgX FP32 precond): {len(hist)-1} cycles, "
              f"final rel. res = {hist[-1]:.2e}")
    except Exception as exc:
        print(f"AmgX path skipped ({exc}). cast/SpMV listings are still valid.")


if __name__ == "__main__":
    _demo()