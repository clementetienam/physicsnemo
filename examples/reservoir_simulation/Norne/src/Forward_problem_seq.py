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

=====================================================================
 NVIDIA PHYSICSNEMO SURROGATE RESERVOIR SIMULATION FORWARD MODELLING
 (SEQUENTIAL PROCESSING VERSION)
=====================================================================

This module implements sequential processing for reservoir simulation forward modelling
using NVIDIA PhyNeMo. It provides a machine learning framework for predicting
reservoir behavior using neural networks with sequential processing capabilities.

Key Features:
- Sequential processing for reservoir simulations
- Multi-GPU support for distributed training
- Neural network models for pressure and saturation prediction
- Comprehensive model evaluation and visualization
- Integration with MLflow for experiment tracking

Usage:
    python Forward_problem_seq.py --config-path=conf --config-name=DECK_CONFIG

Inputs:
    - Configuration file with model parameters
    - Training data from reservoir simulations
    - Test data for model evaluation

Outputs:
    - Trained neural network models
    - Prediction results with evaluation metrics
    - Visualization plots for model performance

@Author : Clement Etienam
"""

import os
import getpass
import copy
import time
import pickle
import logging
import warnings
import multiprocessing
from datetime import timedelta
from pathlib import Path
from typing import Any

# 🔧 Third-party Libraries
import gzip
import scipy.io as sio
import numpy as np
import numpy.linalg
from omegaconf import DictConfig
from cpuinfo import get_cpu_info
from filelock import FileLock
import torch
import hydra
from hydra.utils import to_absolute_path
import mlflow
import mlflow.tracking

# 🔥 PhyNeMo & ML Libraries
from physicsnemo.launch.logging import (
    LaunchLogger,
    PythonLogger,
)
from physicsnemo.launch.logging.mlflow import initialize_mlflow
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad

# 📦 Local Modules

from forward.binaries_extract import (
    Black_oil_seq,
    train_polynomial_models,
)

from forward.gradients_extract import (
    loss_func,
    loss_func_physics,
    Black_oil_peacemann,
)

from forward.machine_extract import (
    InitializeLoggers,
    check_and_remove_dirs,
)
from forward.simulator import (
    simulation_data_types,
)
from forward.utils.sequential.seq_misc_operation_1 import load_and_setup_training_data
from data_extract.opm_extract_rates import read_compdats2
from forward.gradients_extract import clip_and_convert_to_float32
from forward.utils.sequential.training_function import run_training_loop

torch.cuda.empty_cache()

# 🖥️ Detect GPU
def is_available() -> bool:
    """Check if NVIDIA GPU is available using native Python methods."""
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger("Forward problem")
    f_handler = logging.FileHandler(filename="read_vectors.log", mode="w")
    formatter = logging.Formatter(" %(asctime)s - %(levelname)s - %(message)s")
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)
    logger.setLevel(logging.INFO)
    return logger


def initialize_environment() -> tuple[bool, int, logging.Logger]:
    """Initialize the environment and return GPU availability, operation mode, and logger."""
    logger = setup_logging()

    # Clear GPU cache
    torch.cuda.empty_cache()

    # Log PyTorch and CUDA information
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"cuDNN Version: {torch.backends.cudnn.version()}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")

    # Check GPU availability
    gpu_available = is_available()
    if gpu_available:
        logger.info("GPU Available with CUDA")
        try:
            # import cupy as cp
            operation_mode = 0
        except ImportError:
            operation_mode = 1
    else:
        logger.info("No GPU Available")
        operation_mode = 1

    # Log CPU information
    cpu_info = get_cpu_info()
    logger.info("CPU Info:")
    for key, value in cpu_info.items():
        logger.info(f"\t{key}: {value}")

    warnings.filterwarnings("ignore")
    return gpu_available, operation_mode, logger


@hydra.main(version_base="1.2", config_path="conf", config_name="DECK_CONFIG")
def main(cfg: DictConfig) -> None:
    """Main function for batch forward problem solving."""
    # Initialize environment
    gpu_available, operation_mode, logger = initialize_environment()

    # Initialize simulation data types
    (
        type_dict,
        ecl_extensions,
        dynamic_props,
        ecl_vectors,
        static_props,
        SUPPORTED_DATA_TYPES,
    ) = simulation_data_types()
    sequences = ["pressure", "saturation", "oil", "gas", "peacemann"]
    model_type = "FNO" if cfg.custom.fno_type == "FNO" else "PINO"
    model_paths = [f"../MODELS/{model_type}/checkpoints_{seq}_seq" for seq in sequences]
    checkpoint_dir = "checkpoints_seq"
    if cfg.custom.model_Distributed == 1:
        dist, logger = InitializeLoggers(cfg)
        if dist.rank == 0:
            base_dirs = ["__pycache__/", "../RUNS", "outputs/"]
            directories_to_check = [checkpoint_dir] + base_dirs + model_paths
            check_and_remove_dirs(
                directories_to_check, cfg.custom.file_response, logger
            )
            logger.info(
                "|-----------------------------------------------------------------|"
            )
    else:
        base_dirs = ["__pycache__/", "../RUNS", "outputs/", "mlruns"]
        directories_to_check = [checkpoint_dir] + base_dirs + model_paths
        check_and_remove_dirs(directories_to_check, cfg.custom.file_response, logger)
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        DistributedManager.initialize()
        dist = DistributedManager()
        if "RANK" not in os.environ:
            os.environ["RANK"] = str(dist.rank)
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = str(dist.rank % torch.cuda.device_count())
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            device_id = dist.rank % gpu_count  # Map rank to available GPUs
            torch.cuda.set_device(device_id)
            logger.info(
                f"Process {dist.rank} is using GPU {device_id}: {torch.cuda.get_device_name(device_id)}"
            )
        else:
            logger.info(f"Process {dist.rank} is using CPU")
        initialize_mlflow(
            experiment_name="PhyNeMo-Reservoir Batch Modelling",
            experiment_desc="PhyNeMo launch development",
            run_name="Reservoir bacth forward modelling",
            run_desc="Reservoir batch forward modelling training",
            user_name=getpass.getuser(),
            mode="offline",
        )
        logger = PythonLogger(name=" PhyNeMo Reservoir_Characterisation")
        LaunchLogger.initialize(use_mlflow=cfg.use_mlflow)  # PhyNeMo launch logger
    device = dist.device
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                PHYNEMO RESERVOIR CHARACTERISATION:              |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        if cfg.custom.model_Distributed == 1:
            logger.info(
                "|-----------------------------------------------------------------|"
            )
            logger.info(
                "|                     MULTI GPU USAGE MODEL :                     |"
            )
            logger.info(
                "|-----------------------------------------------------------------|"
            )
        else:
            logger.info(
                "|-----------------------------------------------------------------|"
            )
            logger.info(
                "|                     SINGLE GPU USAGE MODEL :                    |"
            )
            logger.info(
                "|-----------------------------------------------------------------|"
            )
    oldfolder = os.getcwd()
    os.chdir(oldfolder)
    Relperm = int(cfg.custom.Relperm)
    # interest = cfg.custom.interest
    pde_method = int(cfg.custom.pde_method)
    params = {
        "k_rwmax": torch.tensor(0.3),
        "k_romax": torch.tensor(0.9),
        "k_rgmax": torch.tensor(0.8),
        "n": torch.tensor(2.0),
        "p": torch.tensor(2.0),
        "q": torch.tensor(2.0),
        "m": torch.tensor(2.0),
        "Swi": torch.tensor(0.1),
        "Sor": torch.tensor(0.2),
    }
    if cfg.custom.interest == "Yes":
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        folders_to_create = ["../RUNS", "../data"]
        if dist.rank == 0:
            if os.path.isfile(to_absolute_path("../data/conversions.mat")):
                os.remove(to_absolute_path("../data/conversions.mat"))
            for folder in folders_to_create:
                absolute_path = to_absolute_path(folder)
                lock_path = (
                    absolute_path + ".lock"
                )  # Use a lock file for synchronization
                with FileLock(lock_path):  # Only one process will create the directory
                    if Path(absolute_path).exists():
                        logger.info(f"Directory already exists: {absolute_path}")
                    else:
                        os.makedirs(absolute_path, exist_ok=True)
                        logger.info(f"Created directory: {absolute_path}")
        logger.info(
            "|-----------------------------------------------------------------|"
        )
    if cfg.custom.model_type == "FNO":
        if cfg.custom.fno_type == "FNO":
            folders_to_create = [
                "../MODELS/FNO/checkpoints_saturation_seq",
                "../MODELS/FNO/checkpoints_oil_seq",
                "../MODELS/FNO/checkpoints_pressure_seq",
                "../MODELS/FNO/checkpoints_gas_seq",
                "../MODELS/FNO/checkpoints_peacemann_seq",
            ]
        else:
            folders_to_create = [
                "../MODELS/PINO/checkpoints_saturation_seq",
                "../MODELS/PINO/checkpoints_oil_seq",
                "../MODELS/PINO/checkpoints_pressure_seq",
                "../MODELS/PINO/checkpoints_gas_seq",
                "../MODELS/PINO/checkpoints_peacemann_seq",
            ]
    else:
        if cfg.custom.fno_type == "FNO":
            folders_to_create = [
                "../MODELS/TRANSOLVER/checkpoints_saturation_seq",
                "../MODELS/TRANSOLVER/checkpoints_oil_seq",
                "../MODELS/TRANSOLVER/checkpoints_pressure_seq",
                "../MODELS/TRANSOLVER/checkpoints_gas_seq",
                "../MODELS/TRANSOLVER/checkpoints_peacemann_seq",
            ]
        else:
            folders_to_create = [
                "../MODELS/PI-TRANSOLVER/checkpoints_saturation_seq",
                "../MODELS/PI-TRANSOLVER/checkpoints_oil_seq",
                "../MODELS/PI-TRANSOLVER/checkpoints_pressure_seq",
                "../MODELS/PI-TRANSOLVER/checkpoints_gas_seq",
                "../MODELS/PI-TRANSOLVER/checkpoints_peacemann_seq",
            ]            
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        for folder in folders_to_create:
            absolute_path = to_absolute_path(folder)
            lock_path = absolute_path + ".lock"  # Use a lock file for synchronization
            with FileLock(lock_path):  # Only one process will create the directory
                if Path(absolute_path).exists():
                    logger.info(f"Directory already exists: {absolute_path}")
                else:
                    os.makedirs(absolute_path, exist_ok=True)
                    logger.info(f"Created directory: {absolute_path}")
        logger.info(
            "|-----------------------------------------------------------------|"
        )
    nx = cfg.custom.PROPS.nx
    ny = cfg.custom.PROPS.ny
    nz = cfg.custom.PROPS.nz
    file_path = to_absolute_path("../data/conversions.mat")
    file_exists = os.path.isfile(file_path)
    if file_exists:
        mat = sio.loadmat(file_path)
        steppi = int(mat["steppi"])
        # steppi_indices = mat["steppi_indices"].flatten()
        N_ens = int(mat["N_ens"])
    else:
        steppi = cfg.custom.steppi
        # steppi_indices = np.linspace(1, 164, steppi, dtype=int)
        N_ens = cfg.custom.ntrain
    logger.info(f"Rank {dist.rank}: steppi = {steppi}, N_ens = {N_ens}")
    # oldfolder2 = os.getcwd()
    sourc_dir = cfg.custom.file_location
    source_dir = to_absolute_path(sourc_dir)  # ('../simulator_data')
    effective = np.genfromtxt(Path(source_dir) / "actnum.out", dtype="float")
    effective_i = np.reshape(effective, (nx, ny, nz), "F")
    SWOW = torch.tensor(np.array(np.vstack(cfg.custom.WELLSPECS.SWOW), dtype=float)).to(
        device
    )
    SWOG = torch.tensor(np.array(np.vstack(cfg.custom.WELLSPECS.SWOG), dtype=float)).to(
        device
    )
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                 Learning the interpolation machines    :        |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
    params1_swow, params2_swow = train_polynomial_models(SWOW, device)
    params1_swog, params2_swog = train_polynomial_models(SWOW, device)
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|      Converged  Learning the interpolation machines    :        |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
    # Extract fluid properties
    BO = float(cfg.custom.PROPS.BO)
    BW = float(cfg.custom.PROPS.BW)
    UW = float(cfg.custom.PROPS.UW)
    UO = float(cfg.custom.PROPS.UO)
    SWI = float(cfg.custom.PROPS.SWI)
    SWR = float(cfg.custom.PROPS.SWR)
    SGINI = float(cfg.custom.PROPS.SG1)
    CFO = float(cfg.custom.PROPS.CFO)
    p_atm = float(cfg.custom.PROPS.PATM)
    p_bub = float(cfg.custom.PROPS.PB)
    SO1 = float(cfg.custom.PROPS.SO1)
    # Extract bounds
    DZ = torch.tensor(100).to(device)
    RE = torch.tensor(0.2 * 100).to(device)
    Truee1 = np.genfromtxt(Path(source_dir) / "rossmary.GRDECL", dtype="float")
    Trueea = np.reshape(Truee1.T, (nx, ny, nz), "F")
    Trueea = np.reshape(Trueea, (-1, 1), "F")
    Trueea = Trueea * effective.reshape(-1, 1)
    if dist.rank == 0:
        navail = multiprocessing.cpu_count()
        logger.info(f"Available CPU cores: {navail}")
    njobs = max(1, multiprocessing.cpu_count() // 5)  # Ensure at least 1 core is used
    if dist.rank == 0:
        logger.info(f"Using {njobs} cores for parallel processing.")
    sourc_dir = cfg.custom.file_location
    source_dir = to_absolute_path(sourc_dir)  # ('../simulator_data')

    gass, producers, injectors = read_compdats2(
        to_absolute_path(cfg.custom.COMPLETIONS_DATA),
        to_absolute_path(cfg.custom.SUMMARY_DATA),
    )  # filename
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                         PRINT WELLS                           : |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info("gas injectors wells")
        logger.info(f"Gas injectors: {gass}")
        logger.info("producer well")
        logger.info(f"Producers: {producers}")
        logger.info("water injector well")
        logger.info(f"Injectors: {injectors}")
    well_measurements = cfg.custom.well_measurements
    lenwels = len(well_measurements)
    input_variables = cfg.custom.input_properties
    output_variables = cfg.custom.output_properties
    N_pr = len(producers)  # Number of producers
    well_names = [entry[-1] for entry in producers]  # Producer well names
    well_namesg = [entry[-1] for entry in gass]  # gas injectors well names
    well_namesw = [entry[-1] for entry in injectors]  # water injectors well names
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                         PRINT WELL NAMES                      : |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info("producer well names")
        logger.info(f"Producer well names: {well_names}")
        logger.info("gas injectors well names")
        logger.info(f"Gas injector well names: {well_namesg}")
        logger.info("water injector well names")
        logger.info(f"Water injector well names: {well_namesw}")
    try:
        with gzip.open(to_absolute_path("../data/static.pkl.gz"), "rb") as f2:
            mat = pickle.load(f2)
    except (pickle.PickleError, EOFError, FileNotFoundError) as e:
        logger.error(f"Error loading pickle file: {e}")
        raise
    X_data1 = mat
    for key, value in X_data1.items():
        if dist.rank == 0:
            logger.info(f"For key '{key}':")
            logger.info(f"\tContains inf: {np.isinf(value).any()}")
            logger.info(f"\tContains -inf: {np.isinf(-value).any()}")
            logger.info(f"\tContains NaN: {np.isnan(value).any()}")
            logger.info(
                "|-----------------------------------------------------------------|"
            )
    perm_ensemble = X_data1["ensemble"]
    poro_ensemble = X_data1["ensemblep"]
    perm_ensemble = clip_and_convert_to_float32(perm_ensemble)
    poro_ensemble = clip_and_convert_to_float32(poro_ensemble)
    SWI = torch.from_numpy(np.array(SWI)).to(device)
    SGINI = torch.from_numpy(np.array(SGINI)).to(device)
    SO1 = torch.from_numpy(np.array(SO1)).to(device)
    SWR = torch.from_numpy(np.array(SWR)).to(device)
    UW = torch.from_numpy(np.array(UW)).to(device)
    BW = torch.from_numpy(np.array(BW)).to(device)
    UO = torch.from_numpy(np.array(UO)).to(device)
    BO = torch.from_numpy(np.array(BO)).to(device)
    p_bub = torch.from_numpy(np.array(p_bub)).to(device)
    p_atm = torch.from_numpy(np.array(p_atm)).to(device)
    CFO = torch.from_numpy(np.array(CFO)).to(device)
    mat = sio.loadmat(to_absolute_path("../data/conversions.mat"))
    minK = mat["minK"]
    maxK = mat["maxK"]
    maxT = mat["maxT"]
    minP = mat["minP"]
    maxP = mat["maxP"]
    maxQw = mat["maxQW"]
    maxQg = mat["maxQg"]
    maxQ = mat["maxQ"]
    max_inn_fcn = mat["max_inn_fcn"]
    max_out_fcn = mat["max_out_fcn"]
    target_min = 0.01
    target_max = 1
    max_inn_fcnx = torch.from_numpy(max_inn_fcn).to(device)
    max_out_fcnx = torch.from_numpy(max_out_fcn).to(device)

    training_setup = load_and_setup_training_data(
        input_variables,
        output_variables,
        cfg,
        dist,
        N_ens,
        nx,
        ny,
        nz,
        steppi,
        maxP,
        N_pr,
        lenwels,
        effective_i,
    )
    labelled_loader_train = training_setup["labelled_loader_train"]
    labelled_loader_testt = training_setup["labelled_loader_testt"]
    labelled_loader_trainp = training_setup["labelled_loader_trainp"]
    labelled_loader_testtp = training_setup["labelled_loader_testtp"]
    composite_model = training_setup["composite_model"]
    combined_optimizer = training_setup["combined_optimizer"]
    input_keys = training_setup["input_keys"]
    input_keys_peacemann = training_setup["input_keys_peacemann"]
    output_keys_peacemann = training_setup["output_keys_peacemann"]
    output_keys_pressure = training_setup["output_keys_pressure"]
    output_keys_saturation = training_setup["output_keys_saturation"]
    output_keys_gas = training_setup["output_keys_gas"]
    output_keys_oil = training_setup["output_keys_oil"]
    use_epoch = training_setup["use_epoch"]
    neededM = training_setup["neededM"]
    neededMx = training_setup["neededMx"]
    neededMxt = training_setup["neededMxt"]
    MODELS = training_setup["MODELS"]
    MODELS_C = training_setup["MODELS_C"]
    SCHEDULER = training_setup["SCHEDULER"]

    if "PRESSURE" in output_variables:
        surrogate_pressure = MODELS["PRESSURE"]
        optimizer_pressure = MODELS_C["pressure"]
        scheduler_pressure = SCHEDULER["PRESSURE"]
    if "SGAS" in output_variables:
        surrogate_gas = MODELS["SGAS"]
        optimizer_gas = MODELS_C["gas"]
        scheduler_gas = SCHEDULER["SGAS"]
    if "SWAT" in output_variables:
        surrogate_saturation = MODELS["SATURATION"]
        optimizer_saturation = MODELS_C["saturation"]
        scheduler_saturation = SCHEDULER["SATURATION"]
    if "SOIL" in output_variables:
        surrogate_oil = MODELS["SOIL"]
        optimizer_oil = MODELS_C["oil"]
        scheduler_oil = SCHEDULER["SOIL"]
    surrogate_peacemann = MODELS["PEACEMANN"]
    optimizer_peacemann = MODELS_C["peacemann"]
    scheduler_peacemann = SCHEDULER["PEACEMANN"]


    def training_step(
        model,
        inputin,
        inputin_p,
        TARGETS,
        cfg,
        device,
        input_keys,
        output_keys_saturation,
        steppi,
        output_variables,
        training_step_metrics,
        neededM,
        neededMx,
        epoch,
    ):
        # Prepare input tensors
        if cfg.custom.unroll == "TRUE":
            cfg.training.max_steps = 1500
        input_tensor_p = inputin_p["X"]

        # Initialize accumulators
        loss = 0  # will be used only in non-unroll branch for real graph
        metrics_accumulator = {
            f"{var}_loss": 0.0
            for var in ["pressure", "water", "oil", "gas", "peacemann"]
        }
        metrics_accumulator["peacemanned"] = 0.0

        if cfg.custom.fno_type == "PINO":
            pino_metrics = {
                "pressured": 0.0,
                "saturationd": 0.0,
                "gasd": 0.0,
                "peacemanned": 0.0,
            }

        # ---- K-step truncated BPTT config ----
        # K = window length (number of timesteps per backward)
        # If not defined in cfg, default to full unroll (no truncation).
        if cfg.custom.unroll == "TRUE":
            K = getattr(cfg.custom, "K_unroll", steppi)
            if K < 1:
                K = 1
            if K > steppi:
                K = steppi
            loss_value = 0.0  # numeric logging for unroll branch

        if cfg.custom.unroll == "TRUE":
            if cfg.custom.unroll_cost == "AUTO":
                predictions_prev = None  # Initialize predictions_prev
                loss_window = 0.0        # accumulates loss over a K-window

                for x in range(steppi):
                    # --------- build per-timestep inputs ----------
                    if x == 0:
                        inputin_t = {}
                        for k, v in inputin.items():
                            if isinstance(v, torch.Tensor) and v.dim() == 5:
                                # v: (B, T, nz, nx, ny) -> take timestep x and keep dim=1
                                inputin_t[k] = v[:, x:x+1, ...]
                            else:
                                # static or already right shape
                                inputin_t[k] = v
                    else:
                        # Create input using model's previous predictions
                        inputin_t = {
                            "perm": inputin["perm"][:, x:x+1, ...],
                            "poro": inputin["poro"][:, x:x+1, ...],
                            "pini": predictions_prev["pressure"],
                            "sini": predictions_prev["water"],
                            "sgini": predictions_prev["gas"],
                            "soini": predictions_prev["oil"],
                            "fault": inputin["fault"][:, x:x+1, ...],
                            "Q": inputin["Q"][:, x:x+1, ...],
                            "Qg": inputin["Qg"][:, x:x+1, ...],
                            "Qw": inputin["Qw"][:, x:x+1, ...],
                            "dt": inputin["dt"][:, x:x+1, ...],
                            "t": inputin["t"][:, x:x+1, ...],
                        }

                    # --------- build model input ----------
                    if cfg.custom.model_type == "FNO":
                        tensors_ar = [
                            value
                            for value in inputin_t.values()
                            if isinstance(value, torch.Tensor)
                        ]
                        input_temp = torch.cat(tensors_ar, dim=1)
                    else:
                        # === KEY FIX: build input_temp with channels LAST (B, steppi, nz, nx, ny, C) ===
                        vars_for_cat = []
                        for key in input_keys:
                            t = inputin_t[key]  # (B, steppi, nz, nx, ny)
                            t = t.unsqueeze(-1)  # (B, steppi, nz, nx, ny, 1)
                            vars_for_cat.append(t)

                        # input_temp: (B, steppi, nz, nx, ny, C)
                        input_temp = torch.cat(vars_for_cat, dim=-1)

                    target_chunks = {}
                    step_loss = 0.0  # tensor accumulator for this timestep

                    # --------- targets ----------
                    if "PRESSURE" in output_variables:
                        target_chunks["pressure"] = {
                            "pressure": TARGETS["PRESSURE"]["pressure"][:, x:x+1, ...]
                        }
                    if "SWAT" in output_variables:
                        target_chunks["saturation"] = {
                            "water_sat": TARGETS["SATURATION"]["water_sat"][:, x:x+1, ...]
                        }
                    if "SOIL" in output_variables:
                        target_chunks["oil"] = {
                            "oil_sat": TARGETS["OIL"]["oil_sat"][:, x:x+1, ...]
                        }
                    if "SGAS" in output_variables:
                        target_chunks["gas"] = {
                            "gas_sat": TARGETS["GAS"]["gas_sat"][:, x:x+1, ...]
                        }

                    # --------- model predictions ----------
                    predictions = {}
                    if "PRESSURE" in output_variables:
                        predictions["pressure"] = model(input_temp, mode="pressure")[
                            "pressure"
                        ]
                    if "SGAS" in output_variables:
                        predictions["gas"] = model(input_temp, mode="gas")["gas"]
                    if "SWAT" in output_variables:
                        predictions["water"] = model(input_temp, mode="saturation")[
                            "saturation"
                        ]
                    if "SOIL" in output_variables:
                        predictions["oil"] = model(input_temp, mode="oil")["oil"]

                    # --------- supervised losses ----------
                    if "PRESSURE" in output_variables:
                        pressure_loss = loss_func(
                            predictions["pressure"],
                            target_chunks["pressure"]["pressure"],
                            "eliptical",
                            cfg.loss.weights.pressure,
                            p=2.0,
                        )
                        step_loss = step_loss + pressure_loss
                        metrics_accumulator["pressure_loss"] += pressure_loss.item()

                    if "SWAT" in output_variables:
                        water_loss = loss_func(
                            predictions["water"],
                            target_chunks["saturation"]["water_sat"],
                            "hyperbolic",
                            cfg.loss.weights.water_sat,
                            p=2.0,
                        )
                        step_loss = step_loss + water_loss
                        metrics_accumulator["water_loss"] += water_loss.item()

                    if "SOIL" in output_variables:
                        oil_loss = loss_func(
                            predictions["oil"],
                            target_chunks["oil"]["oil_sat"],
                            "hyperbolic",
                            cfg.loss.weights.oil_sat,
                            p=2.0,
                        )
                        step_loss = step_loss + oil_loss
                        metrics_accumulator["oil_loss"] += oil_loss.item()

                    if "SGAS" in output_variables:
                        gas_loss = loss_func(
                            predictions["gas"],
                            target_chunks["gas"]["gas_sat"],
                            "hyperbolic",
                            cfg.loss.weights.gas_sat,
                            p=2.0,
                        )
                        step_loss = step_loss + gas_loss
                        metrics_accumulator["gas_loss"] += gas_loss.item()

                    # --------- PINO physics loss ----------
                    if (
                        cfg.custom.fno_type == "PINO"
                        and epoch % max(1, int(0.01 * cfg.training.max_steps)) == 0
                    ):
                        input_varr = {
                            **{
                                k: v
                                if isinstance(v, torch.Tensor) and v.dim() > 2
                                else v
                                for k, v in inputin_t.items()
                            },
                            "pressure": predictions.get("pressure"),
                            "water_sat": predictions.get("water"),
                            "gas_sat": predictions.get("gas"),
                            "oil_sat": predictions.get("oil"),
                        }

                        evaluate = Black_oil_seq(
                            input_varr,
                            neededM,
                            SWI,
                            SWR,
                            UW,
                            BW,
                            UO,
                            BO,
                            nx,
                            ny,
                            chunk_size,
                            SWOW,
                            SWOG,
                            target_min,
                            target_max,
                            minK,
                            maxK,
                            minP,
                            maxP,
                            p_bub,
                            p_atm,
                            CFO,
                            Relperm,
                            params,
                            pde_method,
                            RE,
                            max_inn_fcn,
                            max_out_fcn,
                            DZ,
                            device,
                            params1_swow,
                            params2_swow,
                            params1_swog,
                            params2_swog,
                            maxQw,
                            maxQg,
                            maxQ,
                            maxT,
                        )

                        f_pressure2 = loss_func_physics(
                            evaluate["pressured"], cfg.loss.weights.pressured
                        )
                        f_water2 = loss_func_physics(
                            evaluate["saturationd"], cfg.loss.weights.saturationd
                        )
                        f_gas2 = loss_func_physics(
                            evaluate["gasd"], cfg.loss.weights.gasd
                        )

                        step_loss = step_loss + f_pressure2 + f_water2 + f_gas2
                        pino_metrics["pressured"] += f_pressure2.item()
                        pino_metrics["saturationd"] += f_water2.item()
                        pino_metrics["gasd"] += f_gas2.item()

                    # --------- accumulate into K-window and do backward when window ends ----------
                    loss_window = loss_window + step_loss
                    is_window_end = ((x + 1) % K == 0) or (x == steppi - 1)

                    if is_window_end:
                        # scale by steppi so overall grad is comparable to averaging over time
                        (loss_window / steppi).backward()
                        loss_value += loss_window.detach().item()
                        loss_window = 0.0
                        # detach state for next window (truncate BPTT here)
                        predictions_prev = {
                            k: v.detach() for k, v in predictions.items()
                        }
                    else:
                        # keep graph within this window
                        predictions_prev = predictions.copy()

            else:
                # ---- unroll_cost != "AUTO": K-window BPTT with your existing AR logic ----
                predictions_prev = None
                loss_window = 0.0
                loss_autoregressive = 0.0  

                for x in range(steppi):
                    inputin_t = {}
                    for k, v in inputin.items():
                        if isinstance(v, torch.Tensor) and v.dim() == 5:
                            # v: (B, T, nz, nx, ny) -> take timestep x and keep dim=1
                            inputin_t[k] = v[:, x:x+1, ...]
                        else:
                            # static or already right shape
                            inputin_t[k] = v

                    if cfg.custom.model_type == "FNO":
                        tensors = [
                            value
                            for value in inputin_t.values()
                            if isinstance(value, torch.Tensor)
                        ]
                        input_temp = torch.cat(tensors, dim=1)
                    else:
                        # === KEY FIX: build input_temp with channels LAST (B, steppi, nz, nx, ny, C) ===
                        vars_for_cat = []
                        for key in input_keys:
                            t = inputin_t[key]  # (B, steppi, nz, nx, ny)
                            t = t.unsqueeze(-1)  # (B, steppi, nz, nx, ny, 1)
                            vars_for_cat.append(t)

                        # input_temp: (B, steppi, nz, nx, ny, C)
                        input_temp = torch.cat(vars_for_cat, dim=-1)

                    nz = input_temp.shape[2]
                    fno_expected_nz = chunk_size = nz
                    num_chunks = 1

                    target_chunks = {}
                    step_loss = 0.0

                    # Extract target chunks
                    if "PRESSURE" in output_variables:
                        target_chunks["pressure"] = {
                            "pressure": TARGETS["PRESSURE"]["pressure"][:, x:x+1, ...]
                        }
                    if "SWAT" in output_variables:
                        target_chunks["saturation"] = {
                            "water_sat": TARGETS["SATURATION"]["water_sat"][:, x:x+1, ...]
                        }
                    if "SOIL" in output_variables:
                        target_chunks["oil"] = {
                            "oil_sat": TARGETS["OIL"]["oil_sat"][:, x:x+1, ...]
                        }
                    if "SGAS" in output_variables:
                        target_chunks["gas"] = {
                            "gas_sat": TARGETS["GAS"]["gas_sat"][:, x:x+1, ...]
                        }

                    # Model predictions
                    predictions = {}
                    if "PRESSURE" in output_variables:
                        predictions["pressure"] = model(input_temp, mode="pressure")[
                            "pressure"
                        ]
                    if "SGAS" in output_variables:
                        predictions["gas"] = model(input_temp, mode="gas")["gas"]
                    if "SWAT" in output_variables:
                        predictions["water"] = model(input_temp, mode="saturation")[
                            "saturation"
                        ]
                    if "SOIL" in output_variables:
                        predictions["oil"] = model(input_temp, mode="oil")["oil"]

                    # Supervised losses
                    if "PRESSURE" in output_variables:
                        pressure_loss = loss_func(
                            predictions["pressure"],
                            target_chunks["pressure"]["pressure"],
                            "eliptical",
                            cfg.loss.weights.pressure,
                            p=2.0,
                        )
                        step_loss = step_loss + pressure_loss
                        metrics_accumulator["pressure_loss"] += pressure_loss.item()

                    if "SWAT" in output_variables:
                        water_loss = loss_func(
                            predictions["water"],
                            target_chunks["saturation"]["water_sat"],
                            "hyperbolic",
                            cfg.loss.weights.water_sat,
                            p=2.0,
                        )
                        step_loss = step_loss + water_loss
                        metrics_accumulator["water_loss"] += water_loss.item()

                    if "SOIL" in output_variables:
                        oil_loss = loss_func(
                            predictions["oil"],
                            target_chunks["oil"]["oil_sat"],
                            "hyperbolic",
                            cfg.loss.weights.oil_sat,
                            p=2.0,
                        )
                        step_loss = step_loss + oil_loss
                        metrics_accumulator["oil_loss"] += oil_loss.item()

                    if "SGAS" in output_variables:
                        gas_loss = loss_func(
                            predictions["gas"],
                            target_chunks["gas"]["gas_sat"],
                            "hyperbolic",
                            cfg.loss.weights.gas_sat,
                            p=2.0,
                        )
                        step_loss = step_loss + gas_loss
                        metrics_accumulator["gas_loss"] += gas_loss.item()

                    # PINO physics loss
                    if (
                        cfg.custom.fno_type == "PINO"
                        and epoch % max(1, int(0.01 * cfg.training.max_steps)) == 0
                    ):
                        input_varr = {
                            **{
                                k: v
                                if isinstance(v, torch.Tensor) and v.dim() > 2
                                else v
                                for k, v in inputin_t.items()
                            },
                            "pressure": predictions.get("pressure"),
                            "water_sat": predictions.get("water"),
                            "gas_sat": predictions.get("gas"),
                            "oil_sat": predictions.get("oil"),
                        }

                        evaluate = Black_oil_seq(
                            input_varr,
                            neededM,
                            SWI,
                            SWR,
                            UW,
                            BW,
                            UO,
                            BO,
                            nx,
                            ny,
                            chunk_size,
                            SWOW,
                            SWOG,
                            target_min,
                            target_max,
                            minK,
                            maxK,
                            minP,
                            maxP,
                            p_bub,
                            p_atm,
                            CFO,
                            Relperm,
                            params,
                            pde_method,
                            RE,
                            max_inn_fcn,
                            max_out_fcn,
                            DZ,
                            device,
                            params1_swow,
                            params2_swow,
                            params1_swog,
                            params2_swog,
                            maxQw,
                            maxQg,
                            maxQ,
                            maxT,
                        )

                        f_pressure2 = loss_func_physics(
                            evaluate["pressured"], cfg.loss.weights.pressured
                        )
                        f_water2 = loss_func_physics(
                            evaluate["saturationd"], cfg.loss.weights.saturationd
                        )
                        f_gas2 = loss_func_physics(
                            evaluate["gasd"], cfg.loss.weights.gasd
                        )

                        step_loss = step_loss + f_pressure2 + f_water2 + f_gas2
                        pino_metrics["pressured"] += f_pressure2.item()
                        pino_metrics["saturationd"] += f_water2.item()
                        pino_metrics["gasd"] += f_gas2.item()

                    # Autoregressive loss (your existing logic)
                    if x > 0 and predictions_prev is not None:
                        input_autoregressive = {
                            "perm": inputin["perm"][:, x:x+1, ...],
                            "poro": inputin["poro"][:, x:x+1, ...],
                            "pini": predictions_prev["pressure"],
                            "sini": predictions_prev["water"],
                            "sgini": predictions_prev["gas"],
                            "soini": predictions_prev["oil"],
                            "fault": inputin["fault"][:, x:x+1, ...],
                            "Q": inputin["Q"][:, x:x+1, ...],
                            "Qg": inputin["Qg"][:, x:x+1, ...],
                            "Qw": inputin["Qw"][:, x:x+1, ...],
                            "dt": inputin["dt"][:, x:x+1, ...],
                            "t": inputin["t"][:, x:x+1, ...],
                        }

                        tensors_ar = [
                            value
                            for value in input_autoregressive.values()
                            if isinstance(value, torch.Tensor)
                        ]
                        input_tensor_ar = torch.cat(tensors_ar, dim=1)

                        predictions_ar = {}
                        if "PRESSURE" in output_variables:
                            predictions_ar["pressure"] = model(
                                input_tensor_ar, mode="pressure"
                            )["pressure"]
                        if "SGAS" in output_variables:
                            predictions_ar["gas"] = model(
                                input_tensor_ar, mode="gas"
                            )["gas"]
                        if "SWAT" in output_variables:
                            predictions_ar["water"] = model(
                                input_tensor_ar, mode="saturation"
                            )["saturation"]
                        if "SOIL" in output_variables:
                            predictions_ar["oil"] = model(
                                input_tensor_ar, mode="oil"
                            )["oil"]

                        predictions = predictions_ar  # Replace with autoregressive predictions

                        autoregressive_timestep_loss = 0
                        if "PRESSURE" in output_variables:
                            pressure_loss_ar = loss_func(
                                predictions["pressure"],
                                target_chunks["pressure"]["pressure"],
                                "eliptical",
                                cfg.loss.weights.pressure,
                                p=2.0,
                            )
                            autoregressive_timestep_loss += pressure_loss_ar
                            metrics_accumulator["pressure_loss"] += pressure_loss_ar.item()

                        if "SWAT" in output_variables:
                            water_loss_ar = loss_func(
                                predictions["water"],
                                target_chunks["saturation"]["water_sat"],
                                "hyperbolic",
                                cfg.loss.weights.water_sat,
                                p=2.0,
                            )
                            autoregressive_timestep_loss += water_loss_ar
                            metrics_accumulator["water_loss"] += water_loss_ar.item()

                        if "SOIL" in output_variables:
                            oil_loss_ar = loss_func(
                                predictions["oil"],
                                target_chunks["oil"]["oil_sat"],
                                "hyperbolic",
                                cfg.loss.weights.oil_sat,
                                p=2.0,
                            )
                            autoregressive_timestep_loss += oil_loss_ar
                            metrics_accumulator["oil_loss"] += oil_loss_ar.item()

                        if "SGAS" in output_variables:
                            gas_loss_ar = loss_func(
                                predictions["gas"],
                                target_chunks["gas"]["gas_sat"],
                                "hyperbolic",
                                cfg.loss.weights.gas_sat,
                                p=2.0,
                            )
                            autoregressive_timestep_loss += gas_loss_ar
                            metrics_accumulator["gas_loss"] += gas_loss_ar.item()

                        loss_autoregressive += autoregressive_timestep_loss
                        step_loss = step_loss + (
                            autoregressive_timestep_loss
                            * cfg.loss.weights.get("autoregressive_weight", 0.1)
                        )

                    # accumulate into K-window and backward
                    loss_window = loss_window + step_loss
                    is_window_end = ((x + 1) % K == 0) or (x == steppi - 1)

                    if is_window_end:
                        (loss_window / steppi).backward()
                        loss_value += loss_window.detach().item()
                        loss_window = 0.0
                        predictions_prev = {
                            k: v.detach() for k, v in predictions.items()
                        }
                    else:
                        predictions_prev = predictions.copy()

        else:
            # ------------------ non-unroll branch: unchanged (single backward outside) ------------------
            if cfg.custom.model_type == "FNO":
                tensors = [
                    value
                    for value in inputin.values()
                    if isinstance(value, torch.Tensor)
                ]
                input_tensor = torch.cat(tensors, dim=1)
            else:
                # === KEY FIX: build input_temp with channels LAST (B, steppi, nz, nx, ny, C) ===
                vars_for_cat = []
                for key in input_keys:
                    t = inputin[key]  # (B, steppi, nz, nx, ny)
                    t = t.unsqueeze(-1)  # (B, steppi, nz, nx, ny, 1)
                    vars_for_cat.append(t)

                # input_temp: (B, steppi, nz, nx, ny, C)
                input_tensor = torch.cat(vars_for_cat, dim=-1)

            nz = input_tensor.shape[2]
            fno_expected_nz = chunk_size = nz
            num_chunks = 1
            input_temp = input_tensor
            target_chunks = {}

            # Extract target chunks
            if "PRESSURE" in output_variables:
                target_chunks["pressure"] = {
                    "pressure": TARGETS["PRESSURE"]["pressure"]
                }
            if "SWAT" in output_variables:
                target_chunks["saturation"] = {
                    "water_sat": TARGETS["SATURATION"]["water_sat"]
                }
            if "SOIL" in output_variables:
                target_chunks["oil"] = {"oil_sat": TARGETS["OIL"]["oil_sat"]}
            if "SGAS" in output_variables:
                target_chunks["gas"] = {"gas_sat": TARGETS["GAS"]["gas_sat"]}

            # Model predictions
            predictions = {}
            if "PRESSURE" in output_variables:
                predictions["pressure"] = model(input_temp, mode="pressure")[
                    "pressure"
                ]
            if "SGAS" in output_variables:
                predictions["gas"] = model(input_temp, mode="gas")["gas"]
            if "SWAT" in output_variables:
                predictions["water"] = model(input_temp, mode="saturation")[
                    "saturation"
                ]
            if "SOIL" in output_variables:
                predictions["oil"] = model(input_temp, mode="oil")["oil"]

            if "PRESSURE" in output_variables:
                pressure_loss = loss_func(
                    predictions["pressure"],
                    target_chunks["pressure"]["pressure"],
                    "eliptical",
                    cfg.loss.weights.pressure,
                    p=2.0,
                )
                loss += pressure_loss
                metrics_accumulator["pressure_loss"] += pressure_loss.item()

            if "SWAT" in output_variables:
                water_loss = loss_func(
                    predictions["water"],
                    target_chunks["saturation"]["water_sat"],
                    "hyperbolic",
                    cfg.loss.weights.water_sat,
                    p=2.0,
                )
                loss += water_loss
                metrics_accumulator["water_loss"] += water_loss.item()

            if "SOIL" in output_variables:
                oil_loss = loss_func(
                    predictions["oil"],
                    target_chunks["oil"]["oil_sat"],
                    "hyperbolic",
                    cfg.loss.weights.oil_sat,
                    p=2.0,
                )
                loss += oil_loss
                metrics_accumulator["oil_loss"] += oil_loss.item()

            if "SGAS" in output_variables:
                gas_loss = loss_func(
                    predictions["gas"],
                    target_chunks["gas"]["gas_sat"],
                    "hyperbolic",
                    cfg.loss.weights.gas_sat,
                    p=2.0,
                )
                loss += gas_loss
                metrics_accumulator["gas_loss"] += gas_loss.item()

            # PINO physics loss
            if (
                cfg.custom.fno_type == "PINO"
                and epoch % max(1, int(0.01 * cfg.training.max_steps)) == 0
            ):
                input_varr = {
                    **{
                        k: v
                        if isinstance(v, torch.Tensor) and v.dim() > 2
                        else v
                        for k, v in inputin.items()
                    },
                    "pressure": predictions.get("pressure"),
                    "water_sat": predictions.get("water"),
                    "gas_sat": predictions.get("gas"),
                    "oil_sat": predictions.get("oil"),
                }

                evaluate = Black_oil_seq(
                    input_varr,
                    neededM,
                    SWI,
                    SWR,
                    UW,
                    BW,
                    UO,
                    BO,
                    nx,
                    ny,
                    chunk_size,
                    SWOW,
                    SWOG,
                    target_min,
                    target_max,
                    minK,
                    maxK,
                    minP,
                    maxP,
                    p_bub,
                    p_atm,
                    CFO,
                    Relperm,
                    params,
                    pde_method,
                    RE,
                    max_inn_fcn,
                    max_out_fcn,
                    DZ,
                    device,
                    params1_swow,
                    params2_swow,
                    params1_swog,
                    params2_swog,
                    maxQw,
                    maxQg,
                    maxQ,
                    maxT,
                )

                f_pressure2 = loss_func_physics(
                    evaluate["pressured"], cfg.loss.weights.pressured
                )
                f_water2 = loss_func_physics(
                    evaluate["saturationd"], cfg.loss.weights.saturationd
                )
                f_gas2 = loss_func_physics(
                    evaluate["gasd"], cfg.loss.weights.gasd
                )

                loss += f_pressure2 + f_water2 + f_gas2
                pino_metrics["pressured"] += f_pressure2.item()
                pino_metrics["saturationd"] += f_water2.item()
                pino_metrics["gasd"] += f_gas2.item()

        # ---- scale / aggregate loss for logging ----
        if cfg.custom.unroll == "TRUE":
            # loss_value already includes all time + physics + AR from K-window backward
            loss = loss_value / steppi

        # ---- Peacemann head ----
        outputs_p = model(input_tensor_p, mode="peacemann")
        peacemann_pred = outputs_p["peacemann"]
        target_peacemann = {
            "Y": TARGETS.get("PEACEMANN", {}).get("Y", torch.zeros_like(peacemann_pred))
        }

        peacemann_loss = loss_func(
            peacemann_pred,
            target_peacemann["Y"],
            "peaceman",
            cfg.loss.weights.Y,
            p=2.0,
        )
        metrics_accumulator["peacemann_loss"] = peacemann_loss.item()

        if cfg.custom.unroll == "TRUE":
            # include Peacemann in grads and logging
            peacemann_loss.backward()
            loss += peacemann_loss.item()
        else:
            loss += peacemann_loss  # keep original behaviour

        # Peacemann physics
        if (
            cfg.custom.fno_type == "PINO"
            and epoch % max(1, int(0.01 * cfg.training.max_steps)) == 0
        ):
            inputs1 = {**inputin_p, "Y": peacemann_pred}
            evaluate = Black_oil_peacemann(
                inputs1,
                UO,
                BO,
                UW,
                BW,
                DZ,
                RE,
                device,
                max_inn_fcnx,
                max_out_fcnx,
                params,
                p_bub,
                p_atm,
                steppi,
                CFO,
                Relperm,
                SWI,
                SWR,
                SWOW,
                SWOG,
                params1_swow,
                params2_swow,
                params1_swog,
                params2_swog,
                N_pr,
                lenwels,
            )

            f_peacemann2 = loss_func_physics(
                evaluate["peacemanned"], cfg.loss.weights.peacemanned
            )
            pino_metrics["peacemanned"] = f_peacemann2.item()

            if cfg.custom.unroll == "TRUE":
                f_peacemann2.backward()
                loss += f_peacemann2.item()
            else:
                loss += f_peacemann2

        # Average metrics
        if cfg.custom.unroll == "TRUE":
            denom = steppi
        else:
            denom = 1

        for key in metrics_accumulator:
            training_step_metrics[key] = metrics_accumulator[key] / (
                denom if key != "peacemann_loss" else 1
            )

        if cfg.custom.fno_type == "PINO":
            for key in pino_metrics:
                training_step_metrics[key] = pino_metrics[key] / (
                    num_chunks if key != "peacemanned" else 1
                )

        return loss
       
        
    @StaticCaptureEvaluateNoGrad(
        model=composite_model, logger=logger, use_amp=False, use_graphs=True
    )
    def validation_step(
        model,
        inputin,
        inputin_p,
        TARGETS,
        cfg,
        device,
        input_keys,
        output_keys_saturation,
        steppi,
        output_variables,
        neededM,
        neededMxt,
        val_step_metrics,
    ):
        # Prepare input tensors
        if cfg.custom.unroll == "TRUE":
            cfg.training.max_steps = 1500
        input_tensor_p = inputin_p["X"]
        # Initialize accumulators
        loss = 0
        metrics_accumulator = {
            f"{var}_loss": 0.0
            for var in ["pressure", "water", "oil", "gas", "peacemann"]
        }
        
        if cfg.custom.unroll=="TRUE":
            predictions_prev = None  # Initialize predictions_prev
            for x in range(steppi):
                inputin_t = {}
                for k, v in inputin.items():
                    if isinstance(v, torch.Tensor) and v.dim() == 5:
                        # v: (B, T, nz, nx, ny) -> take timestep x and keep dim=1
                        inputin_t[k] = v[:, x:x+1, ...]
                    else:
                        # static or already right shape
                        inputin_t[k] = v
                        
                
                if cfg.custom.model_type=="FNO":                   
                    tensors = [
                        value for value in inputin_t.values() if isinstance(value, torch.Tensor)
                    ]
                    input_tensor = torch.cat(tensors, dim=1)
                else:

                    # === KEY FIX: build input_temp with channels LAST (B, steppi, nz, nx, ny, C) ===
                    vars_for_cat = []
                    for key in input_keys:
                        t = inputin_t[key]  # (B, steppi, nz, nx, ny)
                        t = t.unsqueeze(-1)  # (B, steppi, nz, nx, ny, 1)
                        vars_for_cat.append(t)

                    # input_temp: (B, steppi, nz, nx, ny, C)
                    input_tensor = torch.cat(vars_for_cat, dim=-1)                  
                                      
                nz = input_tensor.shape[2]

                fno_expected_nz = chunk_size = nz
                num_chunks = 1

                # Extract chunks
                input_temp = input_tensor
                target_chunks = {}

                # Extract target chunks
                if "PRESSURE" in output_variables:
                    target_chunks["pressure"] = {
                        "pressure": TARGETS["PRESSURE"]["pressure"][:, x:x+1, ...]
                    }
                if "SWAT" in output_variables:
                    target_chunks["saturation"] = {
                        "water_sat": TARGETS["SATURATION"]["water_sat"][:, x:x+1, ...]
                    }
                if "SOIL" in output_variables:
                    target_chunks["oil"] = {
                        "oil_sat": TARGETS["OIL"]["oil_sat"][:, x:x+1, ...]
                    }
                if "SGAS" in output_variables:
                    target_chunks["gas"] = {
                        "gas_sat": TARGETS["GAS"]["gas_sat"][:, x:x+1, ...]
                    }

                # Model predictions
                predictions = {}
                if "PRESSURE" in output_variables:
                    predictions["pressure"] = model(input_temp, mode="pressure")["pressure"]
                if "SGAS" in output_variables:
                    predictions["gas"] = model(input_temp, mode="gas")["gas"]
                if "SWAT" in output_variables:
                    predictions["water"] = model(input_temp, mode="saturation")[
                        "saturation"
                    ]
                if "SOIL" in output_variables:
                    predictions["oil"] = model(input_temp, mode="oil")["oil"]
      
                # Compute losses
                if "PRESSURE" in output_variables:
                    pressure_loss = loss_func(
                        predictions["pressure"],
                        target_chunks["pressure"]["pressure"],
                        "eliptical",
                        cfg.loss.weights.pressure,
                        p=2.0,
                    )
                    loss += pressure_loss
                    metrics_accumulator["pressure_loss"] += pressure_loss.item()

                if "SWAT" in output_variables:
                    water_loss = loss_func(
                        predictions["water"],
                        target_chunks["saturation"]["water_sat"],
                        "hyperbolic",
                        cfg.loss.weights.water_sat,
                        p=2.0,
                    )
                    loss += water_loss
                    metrics_accumulator["water_loss"] += water_loss.item()

                if "SOIL" in output_variables:
                    oil_loss = loss_func(
                        predictions["oil"],
                        target_chunks["oil"]["oil_sat"],
                        "hyperbolic",
                        cfg.loss.weights.oil_sat,
                        p=2.0,
                    )
                    loss += oil_loss
                    metrics_accumulator["oil_loss"] += oil_loss.item()

                if "SGAS" in output_variables:
                    gas_loss = loss_func(
                        predictions["gas"],
                        target_chunks["gas"]["gas_sat"],
                        "hyperbolic",
                        cfg.loss.weights.gas_sat,
                        p=2.0,
                    )
                    loss += gas_loss
                    metrics_accumulator["gas_loss"] += gas_loss.item()
                       
        else:
            if cfg.custom.model_type=="FNO":                   
                tensors = [
                    value for value in inputin.values() if isinstance(value, torch.Tensor)
                ]
                input_tensor = torch.cat(tensors, dim=1)
            else:

                # === KEY FIX: build input_temp with channels LAST (B, steppi, nz, nx, ny, C) ===
                vars_for_cat = []
                for key in input_keys:
                    t = inputin[key]  # (B, steppi, nz, nx, ny)
                    t = t.unsqueeze(-1)  # (B, steppi, nz, nx, ny, 1)
                    vars_for_cat.append(t)

                # input_temp: (B, steppi, nz, nx, ny, C)
                input_tensor = torch.cat(vars_for_cat, dim=-1)                
                                  
            #input_tensor_p = inputin_p["X"]
            nz = input_tensor.shape[2]

            fno_expected_nz = chunk_size = nz
            num_chunks = 1
            # Extract chunks
            input_temp = input_tensor
            target_chunks = {}

            # Extract target chunks
            if "PRESSURE" in output_variables:
                target_chunks["pressure"] = {
                    "pressure": TARGETS["PRESSURE"]["pressure"]
                }
            if "SWAT" in output_variables:
                target_chunks["saturation"] = {
                    "water_sat": TARGETS["SATURATION"]["water_sat"]
                }
            if "SOIL" in output_variables:
                target_chunks["oil"] = {
                    "oil_sat": TARGETS["OIL"]["oil_sat"]
                }
            if "SGAS" in output_variables:
                target_chunks["gas"] = {
                    "gas_sat": TARGETS["GAS"]["gas_sat"]
                }

            # Model predictions
            predictions = {}
            if "PRESSURE" in output_variables:
                predictions["pressure"] = model(input_temp, mode="pressure")["pressure"]
            if "SGAS" in output_variables:
                predictions["gas"] = model(input_temp, mode="gas")["gas"]
            if "SWAT" in output_variables:
                predictions["water"] = model(input_temp, mode="saturation")[
                    "saturation"
                ]
            if "SOIL" in output_variables:
                predictions["oil"] = model(input_temp, mode="oil")["oil"]
      
            if "PRESSURE" in output_variables:
                pressure_loss = loss_func(
                    predictions["pressure"],
                    target_chunks["pressure"]["pressure"],
                    "eliptical",
                    cfg.loss.weights.pressure,
                    p=2.0,
                )
                loss += pressure_loss
                metrics_accumulator["pressure_loss"] += pressure_loss.item()

            if "SWAT" in output_variables:
                water_loss = loss_func(
                    predictions["water"],
                    target_chunks["saturation"]["water_sat"],
                    "hyperbolic",
                    cfg.loss.weights.water_sat,
                    p=2.0,
                )
                loss += water_loss
                metrics_accumulator["water_loss"] += water_loss.item()

            if "SOIL" in output_variables:
                oil_loss = loss_func(
                    predictions["oil"],
                    target_chunks["oil"]["oil_sat"],
                    "hyperbolic",
                    cfg.loss.weights.oil_sat,
                    p=2.0,
                )
                loss += oil_loss
                metrics_accumulator["oil_loss"] += oil_loss.item()

            if "SGAS" in output_variables:
                gas_loss = loss_func(
                    predictions["gas"],
                    target_chunks["gas"]["gas_sat"],
                    "hyperbolic",
                    cfg.loss.weights.gas_sat,
                    p=2.0,
                )
                loss += gas_loss
                metrics_accumulator["gas_loss"] += gas_loss.item()


                
        if cfg.custom.unroll=="TRUE":  
            loss = loss/steppi
            
        outputs_p = model(input_tensor_p, mode="peacemann")
        peacemann_pred = outputs_p["peacemann"]
        target_peacemann = {
            "Y": TARGETS.get("PEACEMANN", {}).get("Y", torch.zeros_like(peacemann_pred))
        }

        peacemann_loss = loss_func(
            peacemann_pred, target_peacemann["Y"], "peaceman", cfg.loss.weights.Y, p=2.0
        )
        loss += peacemann_loss
        metrics_accumulator["peacemann_loss"] = peacemann_loss.item()

        if cfg.custom.unroll == "TRUE":
            denom = steppi
        else:
            denom = 1

        for key in metrics_accumulator:
            val_step_metrics[key] = metrics_accumulator[key] / (
                denom if key != "peacemann_loss" else 1
            )

        return loss
        

    training_step_metrics = {}
    val_step_metrics = {}

    if "PRESSURE" in output_variables:
        best_pressure = copy.deepcopy(surrogate_pressure)
    if "SGAS" in output_variables:
        best_gas = copy.deepcopy(surrogate_gas)
    best_peacemann = copy.deepcopy(surrogate_peacemann)
    if "SWAT" in output_variables:
        best_saturation = copy.deepcopy(surrogate_saturation)
    if "SOIL" in output_variables:
        best_oil = copy.deepcopy(surrogate_oil)    
    start_time = time.time()
    run_training_loop(
        dist,
        logger,
        cfg,
        mlflow,
        use_epoch,
        output_variables,
        surrogate_pressure,
        surrogate_gas,
        surrogate_saturation,
        surrogate_oil,
        surrogate_peacemann,
        labelled_loader_train,
        labelled_loader_trainp,
        labelled_loader_testt,
        labelled_loader_testtp,
        composite_model,
        input_keys,
        input_keys_peacemann,
        output_keys_pressure,
        output_keys_gas,
        output_keys_saturation,
        output_keys_oil,
        output_keys_peacemann,
        training_step,
        validation_step,
        training_step_metrics,
        val_step_metrics,
        steppi,
        UO,
        BO,
        UW,
        BW,
        DZ,
        RE,
        max_inn_fcnx,
        max_out_fcnx,
        params,
        p_bub,
        p_atm,
        CFO,
        Relperm,
        SWI,
        SWR,
        SWOW,
        SWOG,
        params1_swow,
        params2_swow,
        params1_swog,
        params2_swog,
        N_pr,
        lenwels,
        neededM,
        neededMx,
        neededMxt,
        nx,
        ny,
        nz,
        target_min,
        target_max,
        minK,
        maxK,
        minP,
        maxP,
        pde_method,
        max_inn_fcn,
        max_out_fcn,
        scheduler_pressure,
        scheduler_saturation,
        scheduler_oil,
        scheduler_gas,
        scheduler_peacemann,
        optimizer_pressure,
        optimizer_saturation,
        optimizer_oil,
        optimizer_gas,
        optimizer_peacemann,
        best_pressure,
        best_gas,
        best_saturation,
        best_oil,
        best_peacemann,
    )

    if dist.rank == 0:
        mlflow.end_run()
        text = "  Training Converged   "
        logger.info(text)
        logger.info("")
        elapsed_time_secs2 = time.time() - start_time
        msg = (
            "Reservoir Modelling training with Nvidia PhyNeMo took: %s secs (Wall clock time)"
            % timedelta(seconds=round(elapsed_time_secs2))
        )
        logger.info(msg)
        logger.info("")


if __name__ == "__main__":
    main()

