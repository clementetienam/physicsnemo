"""Tests for the ``pushd`` working-directory context manager.

The point of ``pushd`` is to make the change-directory pattern
exception-safe, so we explicitly verify that the previous cwd is restored
even when the ``with`` body raises.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Allow imports from src/ when running pytest from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.path_utils import pushd


def test_pushd_changes_and_restores_cwd(tmp_path):
    before = os.getcwd()
    with pushd(tmp_path):
        # Inside the with block, cwd is the resolved target.
        assert Path(os.getcwd()).resolve() == tmp_path.resolve()
    # On normal exit, cwd is back to where we were.
    assert os.getcwd() == before


def test_pushd_yields_resolved_path(tmp_path):
    with pushd(tmp_path) as p:
        assert isinstance(p, Path)
        assert p == tmp_path.resolve()


def test_pushd_restores_cwd_on_exception(tmp_path):
    """The cwd-leak bug class: pushd must restore cwd even when the block raises."""
    before = os.getcwd()

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom), pushd(tmp_path):
        raise _Boom("simulated failure inside the with-block")

    assert os.getcwd() == before


def test_pushd_accepts_str_path(tmp_path):
    before = os.getcwd()
    with pushd(str(tmp_path)):
        assert Path(os.getcwd()).resolve() == tmp_path.resolve()
    assert os.getcwd() == before


def test_pushd_nested(tmp_path):
    """Nested pushd calls restore in LIFO order."""
    inner = tmp_path / "inner"
    inner.mkdir()
    before = os.getcwd()
    with pushd(tmp_path):
        cwd_outer = os.getcwd()
        with pushd(inner):
            assert Path(os.getcwd()).resolve() == inner.resolve()
        # After the inner block exits, we're back to the outer pushd target.
        assert os.getcwd() == cwd_outer
    assert os.getcwd() == before
