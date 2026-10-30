#!/usr/bin/env python3
"""
Build/run the HYPRE + cuSolver CPR GPU benchmark.

Requires:
  source ./hypre_env.sh
  HYPRE_FP64_DIR and HYPRE_FP32_DIR must be set.

This builds the same CUDA source twice:
  fp64  executable: FP64 outer + FP64 HYPRE pressure AMG + FP64 block smoother
  mixed executable: FP64 outer + FP32 HYPRE pressure AMG + FP32 block smoother
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "hypre_cusolver_mixed_cpr.cu"


def lib_dir(prefix: str) -> Path:
    p = Path(prefix)
    for name in ("lib", "lib64"):
        q = p / name
        if q.exists():
            return q
    raise RuntimeError(f"cannot find lib/lib64 under {prefix}")


def compiler() -> str:
    if os.environ.get("MPICXX"):
        return os.environ["MPICXX"]
    opal = os.environ.get("OPAL_PREFIX")
    if opal and (Path(opal) / "bin" / "mpicxx").exists():
        return str(Path(opal) / "bin" / "mpicxx")
    return "mpicxx"


def compile_one(hypre_dir: str, out_exe: Path, use_fp32_precond: bool,
                emulate_fp64: bool = False) -> None:
    out_exe.parent.mkdir(parents=True, exist_ok=True)
    inc = Path(hypre_dir) / "include"
    lib = lib_dir(hypre_dir)
    cmd = [
        compiler(),
        "-cuda",
        "-O3",
        "-std=c++17",
        f"-DCPR_USE_FP32_PRECOND={1 if use_fp32_precond else 0}",
        f"-DCPR_EMULATE_FP64={1 if emulate_fp64 else 0}",
        str(SRC),
        f"-I{inc}",
        f"-L{lib}",
        f"-Wl,-rpath,{lib}",
        "-lHYPRE",
        "-lcusparse",
        "-lcublas",
        "-lcusolver",
        "-lcudart",
        "-o",
        str(out_exe),
    ]
    print("compile:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def launcher(exe: Path, hypre_dir: str, mode: str):
    env = os.environ.copy()
    ldir = str(lib_dir(hypre_dir))
    env["LD_LIBRARY_PATH"] = ldir + ":" + env.get("LD_LIBRARY_PATH", "")
    opal = env.get("OPAL_PREFIX")
    if opal:
        env["PATH"] = str(Path(opal) / "bin") + ":" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = str(Path(opal) / "lib") + ":" + env["LD_LIBRARY_PATH"]
    if mode == "none":
        return [str(exe)], env
    if mode == "mpirun":
        mpirun = str(Path(opal) / "bin" / "mpirun") if opal else "mpirun"
        return [mpirun, "-np", "1", str(exe)], env
    raise ValueError(mode)


def parse_json(text: str) -> dict:
    m = re.findall(r"\{[\s\S]*\}", text)
    if not m:
        raise RuntimeError("no JSON object in output")
    return json.loads(m[-1])


def run_case(args, exe: Path, side: int, variant: str, hypre_dir: str, log: list[str]) -> dict:
    prefix, env = launcher(exe, hypre_dir, args.launcher)
    cmd = prefix + [
        "--n-side", str(side),
        "--sigma", str(args.sigma),
        "--tol", str(args.tol),
        "--restart", str(args.restart),
        "--max-iters", str(args.max_iters),
        "--amg-print-level", str(args.amg_print_level),
        "--coarsen-type", str(args.coarsen_type),
        "--relax-type", str(args.relax_type),
        "--interp-type", str(args.interp_type),
        "--amg-sweeps", str(args.amg_sweeps),
        "--amg-agg-levels", str(args.amg_agg_levels),
        "--amg-pmax", str(args.amg_pmax),
        "--amg-strong-threshold", str(args.amg_strong_threshold),
        "--amg-trunc-factor", str(args.amg_trunc_factor),
    ]
    if args.progress:
        cmd.append("--progress")
    line = "run: " + " ".join(cmd)
    print(line, flush=True)
    log.append(line + "\n")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env, bufsize=1)
    assert proc.stdout is not None
    out = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        out.append(line)
        log.append(line)
    rc = proc.wait()
    text = "".join(out)
    if rc != 0:
        raise RuntimeError(f"case side={side} failed rc={rc}\n{text}")
    result = parse_json(text)
    result["variant"] = result.get("variant", variant)
    return result


def make_comparison(results: list[dict]) -> list[dict]:
    rows = []
    for side in sorted({r["n_side"] for r in results}):
        fp64 = next((r for r in results if r["n_side"] == side and r["variant"] == "fp64"), None)
        mixed = next((r for r in results if r["n_side"] == side and r["variant"] == "mixed"), None)
        if not fp64 or not mixed:
            continue
        rows.append({
            "n_side": side,
            "cells": fp64["cells"],
            "unknowns": fp64["unknowns"],
            "nnz": fp64["nnz"],
            "pressure_nnz": fp64["pressure_nnz"],
            "fp64_setup_ms": fp64["build_setup_ms"],
            "mixed_setup_ms": mixed["build_setup_ms"],
            "fp64_solve_ms": fp64["solve_ms"],
            "mixed_solve_ms": mixed["solve_ms"],
            "solve_speedup_fp64_over_mixed": fp64["solve_ms"] / mixed["solve_ms"] if mixed["solve_ms"] else None,
            "fp64_total_ms": fp64["total_ms"],
            "mixed_total_ms": mixed["total_ms"],
            "total_speedup_fp64_over_mixed": fp64["total_ms"] / mixed["total_ms"] if mixed["total_ms"] else None,
            "fp64_iters": fp64["iters"],
            "mixed_iters": mixed["iters"],
            "fp64_residual": fp64["final_residual"],
            "mixed_residual": mixed["final_residual"],
            "residual_ratio_mixed_over_fp64": mixed["final_residual"] / fp64["final_residual"] if fp64["final_residual"] else None,
            "fp64_converged": fp64["converged"],
            "mixed_converged": mixed["converged"],
        })
    return rows


def print_comparison(rows: list[dict]) -> None:
    if not rows:
        return
    print("\nFP64 vs mixed HYPRE+cuSolver CPR", flush=True)
    print(f"{'side':>6} {'unknowns':>10} {'fp64 ms':>12} {'mixed ms':>12} {'speedup':>8} {'fp64 res':>11} {'mixed res':>11}", flush=True)
    for r in rows:
        print(f"{r['n_side']:>6} {r['unknowns']:>10} {r['fp64_solve_ms']:>12.3f} {r['mixed_solve_ms']:>12.3f} "
              f"{r['solve_speedup_fp64_over_mixed']:>8.3f} {r['fp64_residual']:>11.2e} {r['mixed_residual']:>11.2e}", flush=True)


def make_plots(payload: dict, out_png: Path) -> None:
    results = payload["results"]
    rows = payload["fp64_vs_mixed"]
    if not rows:
        return

    sides = [r["n_side"] for r in rows]
    fp64 = [r for r in results if r["variant"] == "fp64"]
    mixed = [r for r in results if r["variant"] == "mixed"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("HYPRE + cuSolver GPU CPR: FP64 vs mixed precision", fontsize=16, weight="bold")

    ax = axes[0, 0]
    ax.plot(sides, [r["unknowns"] for r in rows], "o-", label="unknowns")
    ax.plot(sides, [r["nnz"] for r in rows], "s-", label="nnz(A)")
    ax.plot(sides, [r["pressure_nnz"] for r in rows], "^-", label="nnz(Ap)")
    ax.set_yscale("log")
    ax.set_xlabel("grid side")
    ax.set_title("Problem size")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(sides, [r["fp64_solve_ms"] for r in rows], "o-", label="FP64 solve")
    ax.plot(sides, [r["mixed_solve_ms"] for r in rows], "s-", label="mixed solve")
    ax.set_yscale("log")
    ax.set_xlabel("grid side")
    ax.set_ylabel("solve time [ms]")
    ax.set_title("Solve timing")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(sides, [r["fp64_residual"] for r in rows], "o-", label="FP64 residual")
    ax.plot(sides, [r["mixed_residual"] for r in rows], "s-", label="mixed residual")
    ax.axhline(payload["tol"], color="black", ls=":", label=f"tol {payload['tol']:.0e}")
    ax.set_yscale("log")
    ax.set_xlabel("grid side")
    ax.set_ylabel(r"relative residual $\|b-Ax\|/\|b\|$")
    ax.set_title("Accuracy")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(sides, [r["solve_speedup_fp64_over_mixed"] for r in rows], "o-", label="solve speedup")
    ax.plot(sides, [r["total_speedup_fp64_over_mixed"] for r in rows], "s--", label="total speedup")
    ax.axhline(1.0, color="black", ls=":", label="break-even")
    ax.set_xlabel("grid side")
    ax.set_ylabel("FP64 / mixed [x]")
    ax.set_title("Speedup")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

    conv_png = out_png.with_name(out_png.stem + "_convergence.png")
    fig, axes = plt.subplots(len(sides), 1, figsize=(10, max(4, 3 * len(sides))), squeeze=False)
    for ax, side in zip(axes[:, 0], sides):
        f = next((r for r in fp64 if r["n_side"] == side), None)
        m = next((r for r in mixed if r["n_side"] == side), None)
        if f:
            h = f.get("residual_history", [])
            ax.plot(range(len(h)), h, "o-", label="FP64")
        if m:
            h = m.get("residual_history", [])
            ax.plot(range(len(h)), h, "s-", label="mixed")
        ax.axhline(payload["tol"], color="black", ls=":")
        ax.set_yscale("log")
        ax.set_xlabel("restart/check step")
        ax.set_ylabel("relative residual")
        ax.set_title(f"Convergence: {side}^3")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend()
    fig.tight_layout()
    fig.savefig(conv_png, dpi=170)
    plt.close(fig)
    print(f"wrote plot: {out_png}", flush=True)
    print(f"wrote convergence plot: {conv_png}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="./runs_hypre_cusolver")
    ap.add_argument("--sides", default="64,96,128",
                    help="comma-separated grid sides; runs sequentially in this order")
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--restart", type=int, default=30)
    ap.add_argument("--max-iters", type=int, default=80)
    ap.add_argument("--amg-print-level", type=int, default=0)
    # --- AMG hierarchy: strong, GPU-tuned defaults (see .cu AMGConfig) ---
    ap.add_argument("--coarsen-type", type=int, default=8)      # PMIS (GPU)
    ap.add_argument("--relax-type", type=int, default=18)       # l1-Jacobi
    ap.add_argument("--interp-type", type=int, default=6)       # ext+i
    ap.add_argument("--amg-sweeps", type=int, default=2)        # 2 sweeps/level
    ap.add_argument("--amg-agg-levels", type=int, default=1)    # aggressive coarsening
    ap.add_argument("--amg-pmax", type=int, default=4)
    ap.add_argument("--amg-strong-threshold", type=float, default=0.5)  # 3D
    ap.add_argument("--amg-trunc-factor", type=float, default=0.3)
    ap.add_argument("--emulate", action="store_true",
                    help="also build/run the df64-emulated-FP64 mixed variant "
                         "(CPR_EMULATE_FP64=1) and compare it to true-FP64 mixed")
    ap.add_argument("--launcher", choices=["mpirun", "none"], default="mpirun")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    hypre_fp64_dir = os.environ.get("HYPRE_FP64_DIR")
    hypre_fp32_dir = os.environ.get("HYPRE_FP32_DIR")
    if not hypre_fp64_dir or not hypre_fp32_dir:
        raise RuntimeError("source hypre_env.sh first; HYPRE_FP64_DIR and HYPRE_FP32_DIR must be set")

    outdir = Path(args.logdir).resolve()
    build = outdir / "build"
    outdir.mkdir(parents=True, exist_ok=True)
    exe_fp64 = build / "hypre_cusolver_cpr_fp64"
    exe_mixed = build / "hypre_cusolver_cpr_mixed"
    compile_one(hypre_fp64_dir, exe_fp64, use_fp32_precond=False)
    compile_one(hypre_fp32_dir, exe_mixed, use_fp32_precond=True)
    exe_emu = build / "hypre_cusolver_cpr_mixed_emu"
    if args.emulate:
        # mixed precision but with the FP64 operator work emulated in df64.
        compile_one(hypre_fp32_dir, exe_emu, use_fp32_precond=True, emulate_fp64=True)

    sides = [int(x) for x in args.sides.split(",") if x.strip()]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log = [f"HYPRE+cuSolver mixed CPR run {stamp}\n"]
    results = []
    for side in sides:
        print(f"\n===== HYPRE+cuSolver FP64 CPR side={side} =====", flush=True)
        results.append(run_case(args, exe_fp64, side, "fp64", hypre_fp64_dir, log))
        print(f"\n===== HYPRE+cuSolver mixed CPR side={side} =====", flush=True)
        results.append(run_case(args, exe_mixed, side, "mixed", hypre_fp32_dir, log))
        if args.emulate:
            print(f"\n===== HYPRE+cuSolver mixed-emu (df64) CPR side={side} =====", flush=True)
            results.append(run_case(args, exe_emu, side, "mixed-emu", hypre_fp32_dir, log))

    comparison = make_comparison(results)

    payload = {
        "method": "HYPRE + cuSolver/cuSPARSE/cuBLAS GPU CPR comparison",
        "tol": args.tol,
        "results": results,
        "fp64_vs_mixed": comparison,
    }
    json_path = outdir / f"hypre_cusolver_cpr_compare_{stamp}.json"
    log_path = outdir / f"hypre_cusolver_cpr_compare_{stamp}.log"
    png_path = outdir / f"hypre_cusolver_cpr_compare_{stamp}_summary.png"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_path.write_text("".join(log), encoding="utf-8")
    print_comparison(comparison)
    make_plots(payload, png_path)
    print(f"wrote json: {json_path}", flush=True)
    print(f"wrote log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
