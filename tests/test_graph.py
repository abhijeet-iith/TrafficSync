"""Network layer: Laplacian properties and spectra."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trafficsync.network import corridor, erdos_renyi, grid, load_network
from trafficsync.network.spectral import is_connected

SRC = Path(__file__).resolve().parents[1] / "src" / "trafficsync"
SHORT = {"duration_s": 900, "warmup_s": 300}
PERFECT = {"mode": "perfect_sensor"}


@pytest.mark.parametrize("net", [corridor(3), grid(2, 2), grid(3, 3)])
def test_laplacian_properties(net):
    L = net.L
    assert np.allclose(L @ np.ones(net.N), 0)
    assert np.allclose(L, L.T)
    assert np.linalg.eigvalsh(L).min() >= -1e-9
    assert np.allclose(L, net.B @ net.B.T)
    s = net.spectral_summary()
    assert s["lambda2"] > 0 and s["connected"]


def test_lambda2_positive_iff_connected():
    for seed in range(20):
        A = erdos_renyi(8, 0.25, seed)
        lam2 = np.sort(np.linalg.eigvalsh(np.diag(A.sum(1)) - A))[1]
        assert (lam2 > 1e-9) == is_connected(A)


def test_known_spectra():
    s = grid(3, 3).spectral_summary()
    assert s["lambda_max_A"] == pytest.approx(2 * np.sqrt(2))
    assert s["lambda2"] == pytest.approx(1.0)
    assert load_network("corridor3").N == 3 and load_network("grid3x3").N == 9


# --------------------------------------------------------------- simulator


