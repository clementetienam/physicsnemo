"""
SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
                            MACHINE EXTRACTION
=====================================================================

This module provides machine learning model extraction capabilities for
reservoir simulation forward modeling. It includes functions for creating
FNO models, GNN models, composite models, and managing model operations.

Key Features:
- FNO (Fourier Neural Operator) model creation and management
- GNN (Graph Neural Network) model implementation
- Composite model orchestration
- Model saving and loading utilities
- MLflow integration for experiment tracking

Usage:
    from forward.machine_extract import (
        FNOModel,
        create_fno_model,
        CompositeModel,
        CompositeOptimizer,
        GNNModel,
        create_gnn_model
    )

@Author : Clement Etienam
"""

# 🛠 Standard Library
import os
import time
import logging
import shutil
from typing import Tuple

# 🔧 Third-party Libraries
from omegaconf import DictConfig
import torch

# 📊 MLFlow & Logging
import mlflow
import mlflow.tracking
from mlflow.tracking import MlflowClient

# 🔥 PhyNeMo & ML Libraries
from physicsnemo.models.fno import FNO
from physicsnemo.models.module import Module
from physicsnemo.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from physicsnemo.distributed import DistributedManager
from physicsnemo.models.transolver import Transolver


# 📦 Local Modules


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger("Forward problem")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class FNOModel(Module):
    def __init__(
        self,
        input_dim,
        steppi,
        output_shape,
        device,
        num_layers=4,
        decoder_layers=1,
        decoder_layer_size=32,
        dimension=3,
        latent_channels=32,
        num_fno_layers=4,
        padding=8,
        num_fno_modes=16,
    ):
        super().__init__()
        self.fno = FNO(
            in_channels=input_dim,
            out_channels=output_shape * steppi,
            decoder_layers=decoder_layers,
            decoder_layer_size=decoder_layer_size,
            dimension=dimension,
            latent_channels=latent_channels,
            num_fno_layers=num_fno_layers,
            padding=padding,
            num_fno_modes=num_fno_modes,
        ).to(torch.device(device))  # Explicit device conversion
        self.meta = type("", (), {})()  # Empty object
        self.meta.name = "fno_model"

    def forward(self, x):
        return self.fno(x)


def create_fno_model(
    input_dim,
    steppi,
    output_shape,
    device,
    num_layers=4,
    decoder_layers=1,
    decoder_layer_size=32,
    dimension=3,
    latent_channels=32,
    num_fno_layers=4,
    padding=8,
    num_fno_modes=16,
):
    """
    Create a Fourier Neural Operator (FNO) model wrapped in a compatible PhyNeMo Module.

    Parameters:
    -----------
    input_dim : int
        The number of input features (e.g., the number of spatial points).
    steppi : int
        Number of time steps or resolution in the output.
    output_shape : int
        The number of distinct outputs to predict.
    device : str
        Device to create the model on ('cpu' or 'cuda').
    num_layers : int, optional
        The number of layers in the FNO. Default is 4.
    decoder_layers : int, optional
        The number of decoder layers. Default is 1.
    decoder_layer_size : int, optional
        The size of each decoder layer. Default is 32.
    dimension : int, optional
        The dimensionality of the FNO (2D or 3D). Default is 2.
    latent_channels : int, optional
        Number of latent channels. Default is 32.
    num_fno_layers : int, optional
        Number of FNO layers. Default is 4.
    padding : int, optional
        Padding for the FNO. Default is 8.
    num_fno_modes : int, optional
        Number of Fourier modes to use. Default is 16.

    Returns:
    --------
    fno_model : FNOModel
        Initialized FNO model ready for inference or training.
    """
    # Validate arguments
    if dimension not in [1, 2, 3]:
        raise ValueError(f"Invalid dimension: {dimension}. Must be 1, 2 or 3.")
    return FNOModel(
        input_dim=input_dim,
        steppi=steppi,
        output_shape=output_shape,
        device=device,
        num_layers=num_layers,
        decoder_layers=decoder_layers,
        decoder_layer_size=decoder_layer_size,
        dimension=dimension,
        latent_channels=latent_channels,
        num_fno_layers=num_fno_layers,
        padding=padding,
        num_fno_modes=num_fno_modes,
    )
    


class TransolverModel(Module):
    def __init__(
        self,
        functional_dim,
        out_dim,
        device,
        embedding_dim=None,
        n_layers=8,
        n_hidden=256,
        dropout=0.0,
        n_head=8,
        act="gelu",
        mlp_ratio=4,
        slice_num=32,
        unified_pos=True,
        ref=8,
        structured_shape=(46, 112),
        use_te=True,
        time_input=False,
    ):
        super().__init__()
        self.transolver = Transolver(
            functional_dim=functional_dim,
            out_dim=out_dim,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout=dropout,
            n_head=n_head,
            act=act,
            mlp_ratio=mlp_ratio,
            slice_num=slice_num,
            unified_pos=unified_pos,
            ref=ref,
            structured_shape=structured_shape,
            use_te=use_te,
            time_input=time_input,
        ).to(torch.device(device))  # Explicit device conversion
        self.meta = type("", (), {})()  # Empty object
        self.meta.name = "transolver_model"

    def forward(self, x):
        return self.transolver(x)


def create_transolver_model(
    functional_dim,
    out_dim,
    device,
    embedding_dim=None,
    n_layers=8,
    n_hidden=16,
    dropout=0.0,
    n_head=8,
    act="gelu",
    mlp_ratio=4,
    slice_num=16,
    unified_pos=True,
    ref=8,
    structured_shape=(46, 112),
    use_te=True,
    time_input=False,
):
    """
    Create a Transolver model wrapped in a compatible PhyNeMo Module.

    Parameters:
    -----------
    functional_dim : int
        The dimension of the input values, not including any embeddings.
    out_dim : int
        The dimension of the output of the model.
    device : str
        Device to create the model on ('cpu' or 'cuda').
    embedding_dim : int | None, optional
        The spatial dimension of the input data embeddings. Default is None.
    n_layers : int, optional
        The number of transformer PhysicsAttention layers. Default is 8.
    n_hidden : int, optional
        The hidden dimension of the transformer. Default is 256.
    dropout : float, optional
        The dropout rate. Default is 0.0.
    n_head : int, optional
        The number of attention heads. Default is 8.
    act : str, optional
        The activation function. Default is "gelu".
    mlp_ratio : int, optional
        The ratio of hidden dimension in the MLP. Default is 4.
    slice_num : int, optional
        The number of slices in the PhysicsAttention layers. Default is 32.
    unified_pos : bool, optional
        Whether to use unified positional embeddings. Default is True.
    ref : int, optional
        The reference dimension size when using unified positions. Default is 8.
    structured_shape : tuple, optional
        The shape of the latent space for structured data. Default is (46, 112).
    use_te : bool, optional
        Whether to use transformer engine backend. Default is True.
    time_input : bool, optional
        Whether to include time embeddings. Default is False.

    Returns:
    --------
    transolver_model : TransolverModel
        Initialized Transolver model ready for inference or training.
    """
    # Validate arguments
    if n_hidden % n_head != 0:
        raise ValueError(f"n_hidden ({n_hidden}) must be divisible by n_head ({n_head})")
    
    if unified_pos and structured_shape is None:
        raise ValueError("structured_shape must be provided when unified_pos=True")
    
    if structured_shape is not None and len(structured_shape) not in [2, 3]:
        raise ValueError(f"structured_shape must be 2D or 3D, got {structured_shape}")

    return TransolverModel(
        functional_dim=functional_dim,
        out_dim=out_dim,
        device=device,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        n_hidden=n_hidden,
        dropout=dropout,
        n_head=n_head,
        act=act,
        mlp_ratio=mlp_ratio,
        slice_num=slice_num,
        unified_pos=unified_pos,
        ref=ref,
        structured_shape=structured_shape,
        use_te=use_te,
        time_input=time_input,
    )

        
        
class CompositeModel(Module):
    def __init__(self, MODELS, output_variables, model_type="FNO"):
        super().__init__()
        self.output_variables = output_variables
        self.model_type = model_type  # Main model type for tracking
        
        # Store models and their types
        self.models = {}
        self.model_types = {}  # Track each model's type individually
        
        if "PRESSURE" in self.output_variables:
            self.surrogate_pressure = MODELS["PRESSURE"]
            self.models["pressure"] = self.surrogate_pressure
            self.model_types["pressure"] = model_type  # FNO or Transolver
            
        if "SGAS" in self.output_variables:
            self.surrogate_gas = MODELS["SGAS"]
            self.models["gas"] = self.surrogate_gas
            self.model_types["gas"] = model_type
            
        if "SWAT" in self.output_variables:
            self.surrogate_saturation = MODELS["SATURATION"]
            self.models["saturation"] = self.surrogate_saturation
            self.model_types["saturation"] = model_type
            
        if "SOIL" in self.output_variables:
            self.surrogate_oil = MODELS["SOIL"]
            self.models["oil"] = self.surrogate_oil
            self.model_types["oil"] = model_type
            
        self.surrogate_peacemann = MODELS["PEACEMANN"]
        self.models["peacemann"] = self.surrogate_peacemann
        self.model_types["peacemann"] = "FNO"  # Peacemann is always FNO

    def _handle_3d_to_2d(self, input_tensor, model, model_key):
        """Convert 3D input to 2D slices only for Transolver models"""
        model_type = self.model_types.get(model_key, "FNO")
        
        if model_type == "FNO" or input_tensor.dim() != 5:
            # FNO can handle 3D directly, or input is already 2D
            return model(input_tensor)
        else:
            # Transolver needs 2D input - process each sample and z-slice
            B, num_channels, nz, nx, ny = input_tensor.shape
            all_predictions = []
            
            for i in range(B):
                sample = input_tensor[i:i+1]
                
                # Reshape to 2D slices
                x2d = sample.permute(0, 2, 1, 3, 4).contiguous()  # (1, nz, num_channels, nx, ny)
                x2d = x2d.view(1 * nz, num_channels, nx, ny)      # (nz, num_channels, nx, ny)
                x2d = x2d.permute(0, 2, 3, 1).contiguous()        # (nz, nx, ny, num_channels)

                # Forward pass
                pred2d = model(x2d)
                
                # Handle output shape
                if pred2d.dim() == 4 and pred2d.shape[-1] == 1:
                    pred2d = pred2d.permute(0, 3, 1, 2).contiguous()  # (nz, 1, nx, ny)

                # Reshape back to 3D
                pred_sample = pred2d.view(1, nz, -1, nx, ny).permute(0, 2, 1, 3, 4).contiguous()
                all_predictions.append(pred_sample)
            
            return torch.cat(all_predictions, dim=0)

    def forward(self, input_tensor, mode="both", **kwargs):
        """
        Forward pass for the composite model.
        
        Parameters:
        -----------
        input_tensor : torch.Tensor
            Input tensor for the model.
        mode : str
            Which model to use: "pressure", "saturation", "gas", "oil", "peacemann", or "both".
        
        Returns:
        --------
        dict
            Outputs from the selected model(s).
        """
        outputs = {}
        
        if mode in ["pressure", "both"]:
            outputs["pressure"] = self._handle_3d_to_2d(input_tensor, self.surrogate_pressure, "pressure")
        
        if mode in ["gas", "both"]:
            outputs["gas"] = self._handle_3d_to_2d(input_tensor, self.surrogate_gas, "gas")
        
        if mode in ["saturation", "both"]:
            outputs["saturation"] = self._handle_3d_to_2d(input_tensor, self.surrogate_saturation, "saturation")
        
        if mode in ["oil", "both"]:
            outputs["oil"] = self._handle_3d_to_2d(input_tensor, self.surrogate_oil, "oil")
        
        if mode in ["peacemann", "both"]:
            outputs["peacemann"] = self._handle_3d_to_2d(input_tensor, self.surrogate_peacemann, "peacemann")

        return outputs


class CompositeOptimizer:
    def __init__(self, optimizers):
        """
        Initialize with a dictionary of optimizers.
        Args:
            optimizers (dict): Dictionary with keys as model parts and values as optimizers.
        """
        self.optimizers = optimizers

    def zero_grad(self, set_to_none=True):
        """
        Zero gradients for all optimizers.
        Args:
            set_to_none (bool): Whether to set gradients to None.
        """
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        """Step all optimizers."""
        for optimizer in self.optimizers.values():
            optimizer.step()

    def state_dict(self):
        """Return state_dict for all optimizers."""
        return {key: opt.state_dict() for key, opt in self.optimizers.items()}

    def load_state_dict(self, state_dict):
        """Load state_dict for all optimizers."""
        for key, opt in self.optimizers.items():
            opt.load_state_dict(state_dict[key])


def safe_rmtree(path, retries=3, delay=1):
    for attempt in range(retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
                logger = setup_logging()
                logger.info(f"Removed folder: {path}")
            else:
                logger.info(f"Folder does not exist: {path}")
            break
        except OSError as e:
            if "Device or resource busy" in str(e):
                logger = setup_logging()
                logger.info(f"Resource busy: {path}. Retrying...")
                time.sleep(delay)
            else:
                logger.info(f"Failed to remove {path}: {e}")
    else:
        logger = setup_logging()
        logger.info(f"Failed to remove directory: {path} after {retries} attempts.")


def save_model_to_buffer(model, model_name):
    torch.save(model.state_dict(), model_name)


def write_buffers_to_disk(model_buffers):
    for path, buffer in model_buffers.items():
        buffer.seek(0)  # Reset buffer position before writing
        with open(path, "wb") as f:
            f.write(buffer.read())  # Write buffer content to file
        logger = setup_logging()
        logger.info(f"✅ Model buffer written to disk: {path}")


def InitializeLoggers(cfg: DictConfig) -> Tuple[DistributedManager, PythonLogger]:
    """
    Initializes loggers and distributed manager.

    Parameters
    ----------
    cfg : DictConfig
        Config file parameters.

    Returns
    -------
    Tuple[DistributedManager, PythonLogger]
    """
    # Initialize distributed manager
    DistributedManager.initialize()
    dist_manager = DistributedManager()
    if "RANK" not in os.environ:
        os.environ["RANK"] = str(dist_manager.rank)
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(dist_manager.rank % torch.cuda.device_count())
    logger = setup_logging()
    logger.info(
        f"Process {os.getenv('RANK')}: torch.cuda.device_count() = {torch.cuda.device_count()}"
    )
    logger.info(
        f"Process {os.getenv('RANK')}: Visible GPUs = {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}"
    )
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        device_id = dist_manager.rank % gpu_count  # Map rank to available GPUs
        torch.cuda.set_device(device_id)
        logger.info(
            f"Process {dist_manager.rank} is using GPU {device_id}: {torch.cuda.get_device_name(device_id)}"
        )
    else:
        logger.info(f"Process {dist_manager.rank} is using CPU")
    logger = PythonLogger(name=f"PhyNeMo Reservoir_Characterisation{dist_manager.rank}")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    experiment_name = "PhyNeMo-Reservoir Modelling"
    experiment_id = None
    if dist_manager.rank == 0:
        try:
            tracking_dir = os.path.join(os.getcwd(), "mlruns")
            if os.path.exists(tracking_dir):
                logger = setup_logging()
                logger.info(f"Removing existing directory: {tracking_dir}")
                os.system(f"rm -rf {tracking_dir}")  # Force delete using shell command
                # shutil.rmtree(tracking_dir)  # Remove directory and all contents
            else:
                os.makedirs(tracking_dir, exist_ok=True)
            os.environ["MLFLOW_TRACKING_URI"] = f"file://{tracking_dir}"
            mlflow.set_tracking_uri(f"file://{tracking_dir}")
            # ✅ **Fix: Ensure MlflowClient is correctly used**
            client = MlflowClient()
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger = setup_logging()
                logger.info(f"[MLflow] Creating new experiment: {experiment_name}")
                experiment_id = client.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
                logger = setup_logging()
                logger.info(f"[MLflow] Using existing experiment ID: {experiment_id}")
            mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name="PhyNeMo-Training")
            logger = setup_logging()
            logger.info(
                f"[MLflow] Started run for experiment '{experiment_name}' with ID {experiment_id}"
            )
        except Exception as e:
            logger = setup_logging()
            logger.error(f"Failed to initialize MLFlow on rank 0: {e}")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return dist_manager, RankZeroLoggingWrapper(logger, dist_manager)


def are_models_equal(model1, model2):
    return all(
        torch.equal(param1, param2)
        for param1, param2 in zip(
            model1.state_dict().values(), model2.state_dict().values()
        )
    )


def remove_ddp(model):
    return (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )


def check_and_remove_dirs(directories, response, logger):
    """
    Checks if directories exist and prompts the user for removal.

    Args:
        directories (list): List of directory paths to check and remove.
    """
    for directory in directories:
        if os.path.exists(directory) and os.path.isdir(directory):
            # response = input(f"Directory '{directory}' exists. Do you want to remove it? (yes/no): ").strip().lower()
            if response == "yes":
                shutil.rmtree(directory)
                logger = setup_logging()
                logger.info(f"Removed: {directory}")
            else:
                logger.info(f"Skipped: {directory}")
        else:
            logger.info(f"Directory '{directory}' does not exist.")


def on_rm_error(func, path, exc_info):
    """Error handler for shutil.rmtree() to handle permission issues."""
    logger = setup_logging()
    logger.warning(f"Error removing {path}. Retrying...")
    try:
        os.chmod(path, 0o777)  # Change permissions to ensure it can be deleted
        func(path)  # Retry removing
    except Exception as e:
        logger = setup_logging()
        logger.error(f"Failed to remove {path}: {e}")
