#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""cond_estimate.py -- how well conditioned is the CPR pressure matrix?

Reads the op_Ap.bin (and op_J.bin, if present) that DUMPOP writes and reports

  * the 1-norm condition estimate  ||A||_1 ||A^-1||_1  (Hager/Higham, via a
    sparse LU; cheap and reliable),
  * the 2-norm condition  sigma_max / sigma_min  where the matrix is small
    enough for sparse SVD iterations to finish,
  * the product with the FP32 and FP16 unit roundoffs, which is the number the
    "FP32 is enough for the hierarchy" argument rests on.

    python3 cond_estimate.py OUTPUT_NORNE/op_Ap.bin [--json out.json]

Needs numpy and scipy.  Without them it says so and exits 0, so a missing
package never fails a reproduction run.
"""
import json
import struct
import sys

U32, U16 = 2.0 ** -24, 2.0 ** -11


def read_ap(path):
    import numpy as np
    with open(path, "rb") as f:
        cells, pnnz, pb = struct.unpack("<3i", f.read(12))
        rp = np.frombuffer(f.read(4 * (cells + 1)), dtype="<i4")
        ci = np.frombuffer(f.read(4 * pnnz), dtype="<i4")
        dt = "<f4" if pb == 4 else "<f8"
        pv = np.frombuffer(f.read(pb * pnnz), dtype=dt).astype(np.float64)
    return cells, rp, ci, pv, pb


def read_j(path):
    import numpy as np
    with open(path, "rb") as f:
        n, n_ext, nnz, bs, cells = struct.unpack("<5i", f.read(20))
        rp = np.frombuffer(f.read(4 * (n + 1)), dtype="<i4")
        ci = np.frombuffer(f.read(4 * nnz), dtype="<i4")
        av = np.frombuffer(f.read(8 * nnz), dtype="<f8")
    return n, n_ext, rp, ci, av


def estimates(A, name):
    import numpy as np
    import scipy.sparse.linalg as sla
    out = {"matrix": name, "rows": int(A.shape[0]), "nnz": int(A.nnz)}
    lu = sla.splu(A.tocsc())
    Ainv = sla.LinearOperator(A.shape, matvec=lu.solve,
                              rmatvec=lambda v: lu.solve(v, trans="T"))
    n1 = sla.onenormest(A)
    n1i = sla.onenormest(Ainv)
    out["cond1_est"] = float(n1 * n1i)
    # 2-norm: largest singular value by svds; smallest through the inverse,
    # i.e. the largest singular value of A^-1.  Skipped on very large systems.
    if A.shape[0] <= 400_000:
        try:
            smax = sla.svds(A, k=1, return_singular_vectors=False)[0]
            sinv = sla.svds(Ainv, k=1, return_singular_vectors=False)[0]
            out["sigma_max"] = float(smax)
            out["sigma_min"] = float(1.0 / sinv)
            out["cond2"] = float(smax * sinv)
        except Exception as e:                      # noqa: BLE001
            out["cond2_error"] = str(e)
    c = out.get("cond2", out["cond1_est"])
    out["cond_times_u32"] = c * U32
    out["cond_times_u16"] = c * U16
    # Row scaling: the largest-to-smallest diagonal ratio, which is what an
    # FP16 cast of the unscaled fine level has to fit into its exponent range.
    d = np.abs(A.diagonal())
    d = d[d > 0]
    if d.size:
        out["diag_min"], out["diag_max"] = float(d.min()), float(d.max())
        out["diag_ratio"] = float(d.max() / d.min())
    absv = np.abs(A.data[A.data != 0])
    if absv.size:
        out["abs_min"], out["abs_max"] = float(absv.min()), float(absv.max())
        out["fp16_underflow_entries"] = int((absv < 6.1e-5).sum())
        out["fp16_overflow_entries"] = int((absv > 65504.0).sum())
    return out


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = argv[1]
    jpath = argv[argv.index("--json") + 1] if "--json" in argv else None
    try:
        import numpy as np            # noqa: F401
        import scipy.sparse as sp
    except ImportError:
        print("[cond] numpy/scipy not available; skipped")
        return 0
    res = []
    cells, rp, ci, pv, pb = read_ap(path)
    # Pressure columns are global ids; one rank means global == local.
    Ap = sp.csr_matrix((pv, ci, rp), shape=(cells, cells))
    r = estimates(Ap, "A_p (CPR pressure, %s values)" % ("FP32" if pb == 4 else "FP64"))
    res.append(r)
    jfile = path.replace("op_Ap.bin", "op_J.bin")
    try:
        n, n_ext, jrp, jci, jav = read_j(jfile)
        if n == n_ext:
            J = sp.csr_matrix((jav, jci, jrp), shape=(n, n))
            res.append(estimates(J, "J (coupled Jacobian, FP64)"))
    except FileNotFoundError:
        pass
    for r in res:
        print("[cond] %s: %d rows, cond1 ~ %.3e%s; cond*u32 = %.2e, cond*u16 = %.2e"
              % (r["matrix"], r["rows"], r["cond1_est"],
                 (", cond2 = %.3e" % r["cond2"]) if "cond2" in r else "",
                 r["cond_times_u32"], r["cond_times_u16"]))
        if "diag_ratio" in r:
            print("[cond]   |diagonal| %.3e .. %.3e (ratio %.2e); entries below the "
                  "FP16 normal range %d, above it %d"
                  % (r["diag_min"], r["diag_max"], r["diag_ratio"],
                     r.get("fp16_underflow_entries", 0),
                     r.get("fp16_overflow_entries", 0)))
    if jpath:
        with open(jpath, "w") as f:
            json.dump(res, f, indent=1)
        print("[cond] wrote", jpath)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
