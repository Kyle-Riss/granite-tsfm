# Copyright contributors to the TSFM project
#
import shutil
from pathlib import Path

import pytest

from pipelines.seoul import config
from pipelines.seoul.materialize import materialize_parquets


pytest.importorskip("duckdb")


def test_materialize_writes_parquets(tmp_path: Path):
    if not config.HOURLY_CSV.is_file() or not config.DISTRICT_CSV.is_file() or not config.POPULATION_CSV.is_file():
        pytest.skip("Seoul CSV fixtures not present under Data/")
    out = tmp_path / "out"
    counts = materialize_parquets(artifacts_dir=out)
    assert len(counts) == 3
    assert counts[str(out / config.REGIONAL_HOURLY_PARQUET.name)] > 1000
    assert counts[str(out / config.SEOUL_CITY_HOURLY_PARQUET.name)] > 100
    # Verify readable by pandas
    import pandas as pd

    df = pd.read_parquet(out / config.REGIONAL_HOURLY_PARQUET.name)
    assert "ts" in df.columns and "power" in df.columns and "population" in df.columns
    shutil.rmtree(out, ignore_errors=True)
