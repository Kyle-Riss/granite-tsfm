# Copyright contributors to the TSFM project
#
"""
DuckDB → Parquet materialization for Seoul / regional hourly data.

Pipeline roles
--------------
- regional_hourly_34k.parquet: all regions in ``Data/Temp-pre.csv`` — primary fuel for
  TSFM + GRU (temperature ↔ demand dynamics across regions).
- seoul_city_hourly.parquet: ``region = 서울시`` only — city-level curve for dashboards.
- seoul_district_monthly.parquet: ``Data/seoul_energy_use_pre.csv`` — drill-down and
  persona / baseline adjustments (not required at full resolution for the foundation model).

Hourly rows are joined to ``Data/region_population.csv`` (``population``, 명) on ``region``.

Requires optional deps: ``pip install "granite-tsfm[pipeline]"`` or ``uv sync --extra pipeline``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipelines.seoul.config import (
    ARTIFACTS_DIR,
    DISTRICT_CSV,
    DISTRICT_MONTHLY_PARQUET,
    HOURLY_CSV,
    POPULATION_CSV,
    REGIONAL_HOURLY_PARQUET,
    SEOUL_CITY_HOURLY_PARQUET,
    SEOUL_REGION_LABEL,
)


def _require_duckdb():
    try:
        import duckdb  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "duckdb is required. Install with: uv sync --extra pipeline  " "or pip install 'granite-tsfm[pipeline]'"
        ) from e


def _sql_path(p: Path) -> str:
    """Single-quoted filesystem path for DuckDB SQL (read_csv / COPY / read_parquet)."""
    s = str(Path(p).resolve())
    return "'" + s.replace("'", "''") + "'"


def _sql_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def materialize_parquets(
    hourly_csv: Path | None = None,
    district_csv: Path | None = None,
    population_csv: Path | None = None,
    artifacts_dir: Path | None = None,
) -> dict[str, int]:
    """Load CSVs via DuckDB and write Parquet artifacts. Returns row counts per output."""
    _require_duckdb()
    import duckdb

    hourly_csv = Path(hourly_csv or HOURLY_CSV)
    district_csv = Path(district_csv or DISTRICT_CSV)
    population_csv = Path(population_csv or POPULATION_CSV)
    out_dir = Path(artifacts_dir or ARTIFACTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not hourly_csv.is_file():
        raise FileNotFoundError(f"Hourly CSV not found: {hourly_csv}")
    if not district_csv.is_file():
        raise FileNotFoundError(f"District CSV not found: {district_csv}")
    if not population_csv.is_file():
        raise FileNotFoundError(f"Population CSV not found: {population_csv}")

    con = duckdb.connect(database=":memory:")
    hourly_lit = _sql_path(hourly_csv)
    con.execute(
        f"""
        CREATE OR REPLACE VIEW hourly_raw AS
        SELECT * FROM read_csv_auto({hourly_lit}, header=true, sample_size=-1);
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE hourly_staging AS
        SELECT
          strptime(CAST(datetime AS VARCHAR), '%m/%d/%y %H:%M') AS ts,
          CAST(region AS VARCHAR) AS region,
          CAST(power AS DOUBLE) AS power,
          CAST(temp AS DOUBLE) AS temp,
          CAST(hour AS INTEGER) AS hour,
          CAST(weekday AS INTEGER) AS weekday,
          CAST(month AS INTEGER) AS month,
          CAST(season AS VARCHAR) AS season,
          CAST(is_weekend AS INTEGER) AS is_weekend,
          try_cast(temp_source AS VARCHAR) AS temp_source,
          try_cast(temp_clim AS DOUBLE) AS temp_clim,
          coalesce(try_cast(is_holiday AS INTEGER), 0) AS is_holiday,
          coalesce(try_cast(is_holiday_eve AS INTEGER), 0) AS is_holiday_eve,
          coalesce(try_cast(is_kpx_peak_hour AS INTEGER), 0) AS is_kpx_peak_hour,
          coalesce(try_cast(is_dr_window_proxy AS INTEGER), 0) AS is_dr_window_proxy,
          coalesce(try_cast(is_extreme_heat AS INTEGER), 0) AS is_extreme_heat,
          coalesce(try_cast(is_extreme_cold AS INTEGER), 0) AS is_extreme_cold,
          coalesce(try_cast(is_heatwave_day AS INTEGER), 0) AS is_heatwave_day,
          coalesce(try_cast(is_coldwave_day AS INTEGER), 0) AS is_coldwave_day,
          coalesce(try_cast(seoul_district_cv AS DOUBLE), 0.0) AS seoul_district_cv
        FROM hourly_raw
        WHERE strptime(CAST(datetime AS VARCHAR), '%m/%d/%y %H:%M') IS NOT NULL;
        """
    )

    pop_lit = _sql_path(population_csv)
    con.execute(
        f"""
        CREATE OR REPLACE VIEW population_lookup AS
        SELECT
          CAST(region AS VARCHAR) AS region,
          CAST(population AS BIGINT) AS population
        FROM read_csv_auto({pop_lit}, header=true, sample_size=-1);
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE hourly_enriched AS
        SELECT
          s.*,
          pop.population,
          ROUND(CAST(s.temp AS DOUBLE)) AS temp_bin,
          GREATEST(CAST(s.temp AS DOUBLE) - 18.0, 0) AS cdd,
          GREATEST(18.0 - CAST(s.temp AS DOUBLE), 0) AS hdd,
          CASE
            WHEN CAST(s.temp AS DOUBLE) >= 26 THEN 'cooling'
            WHEN CAST(s.temp AS DOUBLE) <= 10 THEN 'heating'
            ELSE 'mid'
          END AS temp_band
        FROM hourly_staging s
        LEFT JOIN population_lookup pop USING (region);
        """
    )

    regional_out = out_dir / REGIONAL_HOURLY_PARQUET.name
    seoul_out = out_dir / SEOUL_CITY_HOURLY_PARQUET.name
    district_out = out_dir / DISTRICT_MONTHLY_PARQUET.name

    con.execute(
        f"""
        COPY (
          SELECT * FROM hourly_enriched ORDER BY region, ts
        ) TO {_sql_path(regional_out)} (FORMAT PARQUET);
        """
    )

    seoul_q = _sql_str(SEOUL_REGION_LABEL)
    con.execute(
        f"""
        COPY (
          SELECT * FROM hourly_enriched
          WHERE region = {seoul_q}
          ORDER BY ts
        ) TO {_sql_path(seoul_out)} (FORMAT PARQUET);
        """
    )

    district_lit = _sql_path(district_csv)
    con.execute(
        f"""
        CREATE OR REPLACE VIEW district_raw AS
        SELECT * FROM read_csv_auto({district_lit}, header=true, sample_size=-1);
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT
            CAST(district AS VARCHAR) AS district,
            CAST(month AS INTEGER) AS month,
            CAST(usage AS DOUBLE) AS usage
          FROM district_raw
          ORDER BY district, month
        ) TO {_sql_path(district_out)} (FORMAT PARQUET);
        """
    )

    def count_parquet(p: Path) -> int:
        return con.execute(f"SELECT count(*) FROM read_parquet({_sql_path(p)})").fetchone()[0]

    counts = {
        str(regional_out): count_parquet(regional_out),
        str(seoul_out): count_parquet(seoul_out),
        str(district_out): count_parquet(district_out),
    }
    con.close()
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Parquet artifacts from Data/*.csv via DuckDB.")
    parser.add_argument("--hourly-csv", type=Path, default=None, help=f"Default: {HOURLY_CSV}")
    parser.add_argument("--district-csv", type=Path, default=None, help=f"Default: {DISTRICT_CSV}")
    parser.add_argument("--population-csv", type=Path, default=None, help=f"Default: {POPULATION_CSV}")
    parser.add_argument("--artifacts-dir", type=Path, default=None, help=f"Default: {ARTIFACTS_DIR}")
    args = parser.parse_args(argv)

    counts = materialize_parquets(
        hourly_csv=args.hourly_csv,
        district_csv=args.district_csv,
        population_csv=args.population_csv,
        artifacts_dir=args.artifacts_dir,
    )
    for path, n in counts.items():
        print(f"{path}: {n} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
