from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def warehouse_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from sherlock_rx.data.generator import generate

    d = tmp_path_factory.mktemp("wh") / "warehouse"
    generate(d, scale=1.0, seed=7)
    return d


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Use the trained models in models/ if present, else train once for the session."""
    d = ROOT / "models"
    if (d / "intent.pkl").exists():
        return d
    from sherlock_rx.nlp.train import train

    out = tmp_path_factory.mktemp("models")
    train(out, report_dir=tmp_path_factory.mktemp("reports"), log=False)
    return out


@pytest.fixture(scope="session")
def warehouse(warehouse_dir: Path):  # type: ignore[no-untyped-def]
    from sherlock_rx.warehouse.engine import Warehouse

    return Warehouse(warehouse_dir)


@pytest.fixture(scope="session")
def nlu(models_dir: Path):  # type: ignore[no-untyped-def]
    from sherlock_rx.nlp.pipeline import NLU

    return NLU.load(models_dir)


@pytest.fixture()
def bot(warehouse_dir: Path, models_dir: Path):  # type: ignore[no-untyped-def]
    from sherlock_rx.cli import build_bot

    return build_bot(warehouse_dir, models_dir)
