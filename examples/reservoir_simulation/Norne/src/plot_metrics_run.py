"""
SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@Modified by: clement etienam
"""

from pathlib import Path
import math
import pandas as pd
from omegaconf import DictConfig
import numpy as np
import hydra
import matplotlib.pyplot as plt
from physicsnemo.distributed import DistributedManager

def plot_all_metrics(cfg, mlruns_root: Path | str = None):
    """
    Search for the single 'metrics' folder under the given mlruns root,
    read all metric files, and plot one subplot per file.

    Parameters
    ----------
    cfg : object
        Config object that contains cfg.custom.fno_type ("PINO" or "FNO").
    mlruns_root : Path or str, optional
        Root directory that contains the 'mlruns' folder.
        If None, assumes this script is located inside 'src' and mlruns is at src/mlruns.
    """

    def read_metric_file(fp: Path):
        """Read one MLflow metric file (timestamp, value, step)."""
        try:
            df = pd.read_csv(fp, sep=r"\s+", header=None,
                             names=["timestamp", "value", "step"], engine="python")
            df = df[pd.to_numeric(df["step"], errors="coerce").notna() &
                    pd.to_numeric(df["value"], errors="coerce").notna()]
            if df.empty:
                return None
            return df.sort_values("step")[["step", "value"]]
        except Exception:
            return None

    # --- Choose model folder based on cfg ---
    if cfg.custom.model_type == "FNO":
        if cfg.custom.fno_type == "PINO":
            model_dir = Path("../MODELS/PINO") 
        else:
            model_dir = Path("../MODELS/FNO")
    else:
        if cfg.custom.fno_type == "PINO":
            model_dir = Path("../MODELS/PI-TRANSOLVER") 
        else:
            model_dir = Path("../MODELS/TRANSOLVER")        

    # --- Locate mlruns metrics folder ---
    if mlruns_root is None:
        root = Path(__file__).resolve().parent
        mlruns_root = root / "mlruns"
    else:
        mlruns_root = Path(mlruns_root)

    metrics_dirs = list(mlruns_root.rglob("metrics"))
    if not metrics_dirs:
        raise RuntimeError(f"No 'metrics' folder found under: {mlruns_root}")
    metrics_dir = metrics_dirs[0]

    # --- Read all metric files ---
    files = sorted([p for p in metrics_dir.iterdir() if p.is_file()])
    if not files:
        raise RuntimeError(f"No metric files found in: {metrics_dir}")

    n = len(files)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)

    # --- Plot each metric file ---
    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        if i >= n:
            ax.axis("off")
            continue

        fp = files[i]
        df = read_metric_file(fp)
        if df is None:
            ax.text(0.5, 0.5, "unreadable/empty", ha="center", va="center")
            ax.set_title(fp.name, fontsize=9)
            continue

        y = df["value"].to_numpy()
        y_min, y_max = np.min(y), np.max(y)

        # Handle negative or zero values when considering log scale
        positive_min = np.min(y[y > 0]) if np.any(y > 0) else None
        ratio = (y_max / positive_min) if (positive_min and y_max > 0) else None

        ax.plot(df["step"], y, linewidth=1.5)
        ax.set_title(fp.name, fontsize=9)
        ax.set_xlabel("Epoch (step)")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

        # ---- Auto log-scale if range spans > 1e3 ----
        if ratio and ratio > 1e3:
            ax.set_yscale("log")
            ax.set_ylabel("Value (log scale)")
            ax.grid(True, which="both", alpha=0.3)


    plt.tight_layout()

    # --- Save plot image ---
    out_png = model_dir / f"metrics_grid_{cfg.custom.fno_type}.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    
@hydra.main(version_base="1.2", config_path="conf", config_name="DECK_CONFIG")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.rank == 0:
        plot_all_metrics(cfg)


if __name__ == "__main__":
    main()
