import pytest

from omodel import config


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("OMODEL_UNLOCK_HOLDOUT", raising=False)
    config.override(data_dir=tmp_path / "data")
    yield
    config.reset()
