"""Security regression and hardening unit tests."""

import io
import json
import os
import pickle
import sys
import tarfile
import tempfile
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest


# Ensure optional heavy simulation/cosmology dependencies don't block security test execution
for _mod in [
    "cmocean",
    "healpy",
    "illustris_python",
    "illustris_python.util",
    "corner",
    "requests",
    "prospect",
    "prospect.fitting",
    "prospect.io",
    "prospect.likelihood",
    "prospect.models",
    "prospect.sources",
    "prospect.utils",
    "pyslurm",
]:
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            sys.modules[_mod] = MagicMock()

from temet.load.simtxt import _safe_extract_tar  # noqa: E402
from temet.obs.sdss import plotSingleResult  # noqa: E402
from temet.plot.meta import periodic_slurm_status  # noqa: E402
from temet.util.helper import curRepoVersion  # noqa: E402
from temet.util.match import match  # noqa: E402


def test_curRepoVersion_does_not_mutate_cwd():
    """Verify that curRepoVersion does not alter process-wide working directory."""
    cwd_before = os.getcwd()
    version = curRepoVersion()
    cwd_after = os.getcwd()

    assert cwd_before == cwd_after
    assert isinstance(version, str)


def test_periodic_slurm_status_rejects_command_injection():
    """Verify that invalid machine names with shell metacharacters are rejected."""
    with pytest.raises(ValueError, match="Invalid machine name"):
        periodic_slurm_status(machine="vera; echo pwned", nosave=True)

    with pytest.raises(ValueError, match="Invalid machine name"):
        periodic_slurm_status(machine="vera | cat /etc/passwd", nosave=True)


def test_match_strided_array():
    """Verify that match handles non-contiguous strided NumPy slices safely."""
    N = 100
    base_x = np.arange(N, dtype=np.int64)
    # Sliced non-contiguous view
    strided_x = base_x[::2]
    target_y = np.array([10, 20, 30], dtype=np.int64)

    i1, i2 = match(strided_x, target_y)

    assert i1 is not None and i2 is not None
    assert np.array_equal(strided_x[i1], target_y[i2])


def test_safe_extract_tar_traversal_protection():
    """Verify that _safe_extract_tar blocks archives with path traversal entries."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = os.path.join(tmp_dir, "traversal.tar.gz")
        dest_dir = os.path.join(tmp_dir, "dest")
        os.makedirs(dest_dir, exist_ok=True)

        # Create malicious tar in-memory with ../
        with tarfile.open(tar_path, "w:gz") as tar:
            data = b"malicious content"
            tarinfo = tarfile.TarInfo(name="../escape.txt")
            tarinfo.size = len(data)
            tar.addfile(tarinfo, io.BytesIO(data))

        # Extraction must be blocked or filtered safely
        try:
            _safe_extract_tar(tar_path, dest_dir)
            # If tarfile data_filter is active, it strips or alters the path,
            # so escape.txt must NEVER exist outside dest_dir
            assert not os.path.exists(os.path.join(tmp_dir, "escape.txt"))
        except (RuntimeError, tarfile.FilterError):
            # Path traversal exception was raised as expected
            assert not os.path.exists(os.path.join(tmp_dir, "escape.txt"))


def test_sdss_plotSingleResult_pickle_blocked_by_default():
    """Verify that plotSingleResult rejects pickle-encoded model_params by default."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_hdf5 = os.path.join(tmp_dir, "spec_123.hdf5")
        with h5py.File(fake_hdf5, "w") as f:
            f.create_group("sampling")
            f["sampling"]["chain"] = np.zeros((10, 5))
            f.create_group("obs")
            f["obs"]["wavelength"] = np.linspace(4000, 7000, 100)
            f["obs"]["spectrum"] = np.ones(100)
            # Store pickled object (legacy format)
            f.attrs["model_params"] = np.void(pickle.dumps({"theta": [1, 2, 3]}))
            f.attrs["run_params"] = json.dumps({"test": 123})

        with patch("temet.obs.sdss._indivSavePath", return_value=fake_hdf5):
            with pytest.raises(ValueError, match="legacy Python pickle"):
                plotSingleResult(123, allow_pickle=False)
