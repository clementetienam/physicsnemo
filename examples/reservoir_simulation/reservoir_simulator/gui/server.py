#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Local server behind gui/index.html: builds decks, runs the simulator, shows results.

    python3 gui/server.py                  # then open http://127.0.0.1:8765
    python3 gui/server.py --port 9000

Python standard library only.  It listens on 127.0.0.1, or on all interfaces inside
the project container, where the port is published to the host's 127.0.0.1 only.
Every request that starts or stops work must carry the token printed into the page,
so another website open in the same browser cannot start a job here.
"""
import argparse
import json
import mimetypes
import os
import re
import secrets
import shutil
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

GUI = Path(__file__).resolve().parent
ROOT = GUI.parent
CUDA = ROOT / "full_reservoir_simulator" / "GPU_reservoir_simulator" / "cuda"
BENCH = CUDA / "benchmark"
RUNS = BENCH / "runs" / "gui"
TOKEN = secrets.token_urlsafe(24)

# Deck keywords the page may set, each with what the parser accepts.  The parser
# in src/deck.cpp stays the authority: a value it rejects stops the run and the
# message appears in the log.
CHOICES = {
    "LINSOLVER": {"AMGX", "HYPRE", "GINKGO"},
    "GINKGOPREC": {"16", "32"},
    "KRYLOV": {"FGMRES", "BICGSTAB"},
    "MIXED": {"0", "1"},
    "EMULATE": {"0", "1"},
    "BLOCKSPMV": {"0", "1"},
    "CUDAGRAPH": {"0", "1"},
    "AIM": {"0", "1"},
    "JACOBIAN": {"ANALYTIC", "NUMERICAL", "TENSORCORE"},
    "TCPRECISION": {"TF32", "FP32"},
    "TCVALIDATE": {"0", "1"},
}
NUMBERS = {"LINTOL", "LINMAXIT", "TEND", "DTMAX", "NEWTONMAX"}
BINARIES = ("nvrs_cuda", "nvrs_cuda_fp64", "nvrs_cuda_tc", "nvrs_cuda_fp64_tc")
NVRS_COMMANDS = {"install", "cases", "get", "run", "profile", "report", "all"}
NVRS_FLAGS = {"--no-opm", "--no-profile", "--scaling", "--no-scaling"}
BACKENDS = {"amgx", "hypre", "ginkgo", "all", "both"}
CASE_RE = re.compile(r"^[A-Za-z0-9_]{1,40}$")
SERVED = {".html", ".png", ".gif", ".jpg", ".svg", ".pdf", ".json", ".css", ".js", ".md", ".log", ".txt"}

jobs = {}
jobs_lock = threading.Lock()


class Job:
    def __init__(self, label, argv, cwd, outdir, env=None):
        self.id = secrets.token_hex(4)
        self.label = label
        self.argv = argv
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.log = self.outdir / "console.log"
        self.started = time.time()
        self.finished = None
        self.code = None
        self.proc = subprocess.Popen(
            argv, cwd=str(cwd), stdout=open(self.log, "wb"), stderr=subprocess.STDOUT,
            env=env, start_new_session=os.name == "posix")
        threading.Thread(target=self._wait, daemon=True).start()

    def _wait(self):
        self.code = self.proc.wait()
        self.finished = time.time()

    def stop(self):
        if self.code is not None:
            return
        if os.name == "posix":
            os.killpg(self.proc.pid, signal.SIGTERM)
        else:
            self.proc.terminate()

    def info(self):
        return {"id": self.id, "label": self.label, "command": " ".join(self.argv),
                "running": self.code is None, "code": self.code,
                "seconds": round((self.finished or time.time()) - self.started, 1),
                "artifacts": artifacts(self.outdir)}


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def artifacts(folder):
    out = []
    for p in sorted(Path(folder).rglob("*")):
        if p.is_file() and p.suffix.lower() in {".html", ".png", ".gif"}:
            out.append(rel(p))
    return out[:200]


def gpu():
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        return subprocess.run(
            [smi, "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def status():
    reports = sorted(rel(p) for base in (ROOT / "results", BENCH / "runs")
                     if base.exists() for p in base.rglob("*.html"))
    with jobs_lock:
        current = [j.info() for j in jobs.values()]
    return {
        "gpu": gpu(),
        "mpirun": shutil.which(os.environ.get("MPIRUN", "mpirun")) is not None,
        "binaries": {b: (CUDA / b).exists() for b in BINARIES},
        "decks": sorted(p.stem for p in (CUDA / "decks").glob("*.DATA")),
        "reports": reports,
        "jobs": current,
    }


def clean_overrides(overrides):
    lines = []
    for key, value in overrides.items():
        value = str(value).strip()
        if key in CHOICES:
            if value not in CHOICES[key]:
                raise ValueError("%s cannot be %s" % (key, value))
        elif key in NUMBERS:
            float(value)
        else:
            raise ValueError("%s is not offered here" % key)
        lines.append("%s %s" % (key, value))
    return lines


def start_deck_run(body):
    name = body.get("deck", "")
    src = CUDA / "decks" / (name + ".DATA")
    if not CASE_RE.match(name) or not src.exists():
        raise ValueError("no deck named %s" % name)
    binary = body.get("binary", "nvrs_cuda")
    if binary not in BINARIES:
        raise ValueError("unknown binary")
    if not (CUDA / binary).exists():
        raise ValueError("%s is not built yet. Use the Build tab." % binary)
    ranks = int(body.get("ranks", 1))
    if not 1 <= ranks <= 64:
        raise ValueError("ranks must be between 1 and 64")
    extra = clean_overrides(body.get("overrides", {}))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = RUNS / ("%s_%s" % (name, stamp))
    out.mkdir(parents=True, exist_ok=True)
    deck = out / (name + ".DATA")
    text = src.read_text(encoding="utf-8", errors="replace")
    text += "\n-- set by the browser interface\nOUTDIR %s\n%s\n" % (out / "output", "\n".join(extra))
    deck.write_text(text, encoding="utf-8")
    argv = [os.environ.get("MPIRUN", "mpirun"), "-n", str(ranks), str(CUDA / binary), str(deck)]
    return Job("%s, %d rank(s), %s" % (name, ranks, binary), argv, CUDA, out)


def start_nvrs(body):
    command = body.get("command", "")
    if command not in NVRS_COMMANDS:
        raise ValueError("unknown command")
    argv = [str(BENCH / "nvrs.sh"), command]
    case = body.get("case", "")
    if command not in {"install", "cases"}:
        if not CASE_RE.match(case):
            raise ValueError("choose a case")
        argv.append(case)
    for flag in body.get("flags", []):
        if flag not in NVRS_FLAGS:
            raise ValueError("unknown flag %s" % flag)
        argv.append(flag)
    backend = body.get("backend", "")
    if backend:
        if backend not in BACKENDS:
            raise ValueError("unknown backend")
        argv += ["--backend", backend]
    out = RUNS / ("nvrs_%s_%s" % (command, time.strftime("%Y%m%d_%H%M%S")))
    return Job("nvrs.sh " + " ".join(argv[1:]), argv, BENCH, out)


def start_build(body):
    argv = [str(BENCH / "nvrs.sh"), "install"]
    env = dict(os.environ)
    env.pop("TENSORCORE", None)
    if body.get("tensorcore"):
        env["TENSORCORE"] = "1"
    out = RUNS / ("build_%s" % time.strftime("%Y%m%d_%H%M%S"))
    label = ("TENSORCORE=1 " if body.get("tensorcore") else "") + "nvrs.sh install"
    return Job(label, argv, BENCH, out, env)


class Handler(BaseHTTPRequestHandler):
    server_version = "nvrs-gui"
    protocol_version = "HTTP/1.1"

    def handle(self):
        # A browser that leaves the page mid-download is not an error.
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _local_host(self):
        host = self.headers.get("Host", "").split(":")[0]
        return host in {"127.0.0.1", "localhost", "[::1]"}

    def do_GET(self):
        if not self._local_host():
            return self._json({"error": "local requests only"}, 403)
        url = urlparse(self.path)
        if url.path in {"/", "/index.html"}:
            page = (GUI / "index.html").read_text(encoding="utf-8")
            page = page.replace("__NVRS_TOKEN__", TOKEN).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            return self.wfile.write(page)
        if url.path == "/api/status":
            return self._json(status())
        if url.path == "/api/log":
            q = parse_qs(url.query)
            with jobs_lock:
                job = jobs.get(q.get("id", [""])[0])
            if not job:
                return self._json({"error": "no such job"}, 404)
            start = int(q.get("from", ["0"])[0])
            with open(job.log, "rb") as f:
                f.seek(start)
                chunk = f.read(256 * 1024)
            return self._json({"text": chunk.decode("utf-8", "replace"), "next": start + len(chunk),
                               "job": job.info()})
        if url.path.startswith("/files/"):
            target = (ROOT / unquote(url.path[len("/files/"):])).resolve()
            if ROOT not in target.parents or not target.is_file() or target.suffix.lower() not in SERVED:
                return self._json({"error": "not available"}, 404)
            return self._file(target)
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        if not self._local_host():
            return self._json({"error": "local requests only"}, 403)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            return self._json({"error": "bad request"}, 400)
        if not secrets.compare_digest(str(body.get("token", "")), TOKEN):
            return self._json({"error": "bad token: reload the page"}, 403)
        path = urlparse(self.path).path
        try:
            if path == "/api/stop":
                with jobs_lock:
                    job = jobs.get(body.get("id", ""))
                if job:
                    job.stop()
                return self._json({"ok": True})
            starters = {"/api/run": start_deck_run, "/api/nvrs": start_nvrs, "/api/build": start_build}
            if path not in starters:
                return self._json({"error": "not found"}, 404)
            job = starters[path](body)
            with jobs_lock:
                jobs[job.id] = job
            return self._json(job.info())
        except (ValueError, OSError) as e:
            return self._json({"error": str(e)}, 400)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    # Inside the container the published port is what reaches this process.
    host = "0.0.0.0" if os.environ.get("NVRS_IN_CONTAINER") else "127.0.0.1"
    server = ThreadingHTTPServer((host, args.port), Handler)
    print("NVRS-CUDA interface: open http://127.0.0.1:%d   (Ctrl-C stops it)" % args.port, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
