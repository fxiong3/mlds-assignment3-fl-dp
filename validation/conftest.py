import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl.data import make_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def fd_noniid():
    """20 clients, Dirichlet(0.3) label skew, unequal sizes (30-120 samples)."""
    return make_synthetic(num_clients=20, samples_range=(30, 120), noniid_alpha=0.3, seed=1, n_test=1500)


@pytest.fixture(scope="session")
def fd_iid():
    return make_synthetic(num_clients=20, samples_range=(30, 120), noniid_alpha=None, seed=1, n_test=1500)
