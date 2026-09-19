import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
DATA = ROOT / "tests" / "data"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def pcas_tsv():
    return DATA / "datasets_pcas.tsv"


@pytest.fixture(scope="session")
def neighbors_h5():
    return DATA / "datasets_neighbors.h5"


@pytest.fixture(scope="session")
def normalized_h5():
    return DATA / "datasets_normalized_selected.h5"


@pytest.fixture(scope="session", autouse=True)
def _gpu():
    from gpu import setup_gpu
    setup_gpu()
