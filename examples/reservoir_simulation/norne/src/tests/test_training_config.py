"""
Smoke tests for training_config dataclasses.

Verifies that all dataclasses can be instantiated with expected fields and
that optional fields default to None without raising errors.
"""

import sys
import os

# Allow imports from src/ when running pytest from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock

from forward.utils.sequential.training_config import (
    DataLoaders,
    ModelKeys,
    NormParams,
    Optimizers,
    PhysicsParams,
    Schedulers,
    SurrogateModels,
    TrainingState,
)


# ---------------------------------------------------------------------------
# SurrogateModels
# ---------------------------------------------------------------------------

def test_surrogate_models_minimal():
    m = SurrogateModels(composite_model=MagicMock())
    assert m.composite_model is not None
    assert m.surrogate_pressure is None
    assert m.best_peacemann is None


def test_surrogate_models_full():
    mk = MagicMock()
    m = SurrogateModels(
        composite_model=mk,
        surrogate_pressure=mk,
        surrogate_gas=mk,
        surrogate_saturation=mk,
        surrogate_oil=mk,
        surrogate_peacemann=mk,
        best_pressure=mk,
        best_gas=mk,
        best_saturation=mk,
        best_oil=mk,
        best_peacemann=mk,
    )
    assert m.surrogate_peacemann is mk


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def test_data_loaders_required_fields():
    dl = DataLoaders(
        labelled_loader_train=MagicMock(),
        labelled_loader_testt=MagicMock(),
    )
    assert dl.labelled_loader_trainp is None
    assert dl.labelled_loader_testtp is None


def test_data_loaders_all_fields():
    mk = MagicMock()
    dl = DataLoaders(
        labelled_loader_train=mk,
        labelled_loader_testt=mk,
        labelled_loader_trainp=mk,
        labelled_loader_testtp=mk,
    )
    assert dl.labelled_loader_trainp is mk


# ---------------------------------------------------------------------------
# ModelKeys
# ---------------------------------------------------------------------------

def test_model_keys_defaults_empty_lists():
    keys = ModelKeys(
        output_variables=["PRESSURE"],
        input_keys=["inn"],
        input_keys_peacemann=["inn_p"],
    )
    assert keys.output_keys_pressure == []
    assert keys.output_keys_peacemann == []


def test_model_keys_full():
    keys = ModelKeys(
        output_variables=["PRESSURE", "SWAT"],
        input_keys=["a"],
        input_keys_peacemann=["b"],
        output_keys_pressure=["p"],
        output_keys_saturation=["s"],
        output_keys_gas=["g"],
        output_keys_oil=["o"],
        output_keys_peacemann=["pe"],
    )
    assert "p" in keys.output_keys_pressure


# ---------------------------------------------------------------------------
# PhysicsParams
# ---------------------------------------------------------------------------

def test_physics_params_fields():
    mk = MagicMock()
    p = PhysicsParams(
        nx=46, ny=112, nz=22, steppi=10,
        N_pr=8, lenwels=22, neededM=mk, neededMx=mk, neededMxt=mk,
        UO=2.5, BO=1.1, UW=0.5, BW=1.0,
        DZ=100.0, RE=0.007,
        params=mk, p_bub=75.0, p_atm=14.6959, CFO=1e-5,
        Relperm=mk, SWI=0.12, SWR=0.1,
        SWOW=mk, SWOG=mk,
        params1_swow=mk, params2_swow=mk,
        params1_swog=mk, params2_swog=mk,
    )
    assert p.nx == 46
    assert pytest.approx(0.12) == p.SWI


def test_physics_params_pde_method_default_and_override():
    """``pde_method`` was added so step-functions can dispatch a residual flavour."""
    mk = MagicMock()
    common = dict(
        nx=46, ny=112, nz=22, steppi=10,
        N_pr=8, lenwels=22, neededM=mk, neededMx=mk, neededMxt=mk,
        UO=2.5, BO=1.1, UW=0.5, BW=1.0, DZ=100.0, RE=0.007,
        params=mk, p_bub=75.0, p_atm=14.6959, CFO=1e-5,
        Relperm=mk, SWI=0.12, SWR=0.1, SWOW=mk, SWOG=mk,
        params1_swow=mk, params2_swow=mk, params1_swog=mk, params2_swog=mk,
    )
    p_default = PhysicsParams(**common)
    assert p_default.pde_method is None

    p_set = PhysicsParams(**common, pde_method="FDM")
    assert p_set.pde_method == "FDM"


# ---------------------------------------------------------------------------
# NormParams
# ---------------------------------------------------------------------------

def test_norm_params_fields():
    import numpy as np
    n = NormParams(
        max_inn_fcnx=np.ones(5),
        max_out_fcnx=np.ones(5),
        max_inn_fcn=np.ones(5),
        max_out_fcn=np.ones(5),
        target_min=0.0,
        target_max=1.0,
        minK=1.0,
        maxK=2000.0,
        minP=14.0,
        maxP=5000.0,
    )
    assert n.maxK == pytest.approx(2000.0)
    assert n.target_max == pytest.approx(1.0)
    # The added rate/time fields default to None for backward compat with
    # callers that don't pass them.
    assert n.maxT is None
    assert n.maxQ is None
    assert n.maxQw is None
    assert n.maxQg is None


def test_norm_params_rate_and_time_fields():
    """``maxT/Q/Qw/Qg`` were added so the black-oil residuals can denormalise."""
    import numpy as np
    n = NormParams(
        max_inn_fcnx=np.ones(5),
        max_out_fcnx=np.ones(5),
        max_inn_fcn=np.ones(5),
        max_out_fcn=np.ones(5),
        target_min=0.0,
        target_max=1.0,
        minK=1.0,
        maxK=2000.0,
        minP=14.0,
        maxP=5000.0,
        maxT=10000.0,
        maxQ=1e5,
        maxQw=1e4,
        maxQg=1e6,
    )
    assert n.maxT == pytest.approx(10000.0)
    assert n.maxQ == pytest.approx(1e5)
    assert n.maxQw == pytest.approx(1e4)
    assert n.maxQg == pytest.approx(1e6)


# ---------------------------------------------------------------------------
# Optimizers / Schedulers
# ---------------------------------------------------------------------------

def test_optimizers_optional_fields():
    mk = MagicMock()
    opt = Optimizers(optimizer_peacemann=mk)
    assert opt.optimizer_pressure is None
    assert opt.optimizer_gas is None


def test_schedulers_optional_fields():
    mk = MagicMock()
    sch = Schedulers(scheduler_peacemann=mk)
    assert sch.scheduler_pressure is None


# ---------------------------------------------------------------------------
# TrainingState
# ---------------------------------------------------------------------------

def test_training_state_callable_fields():
    def fn():
        return None
    ts = TrainingState(
        training_step=fn,
        validation_step=fn,
        training_step_metrics={},
        val_step_metrics={},
    )
    assert callable(ts.training_step)
    assert ts.training_step_metrics == {}
