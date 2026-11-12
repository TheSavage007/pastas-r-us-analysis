from pathlib import Path

import pandas as pd
import pytest

from src import analysis

DATA = Path(__file__).resolve().parents[1] / "data" / "pastas_r_us_locations.csv"


@pytest.fixture(scope="module")
def df():
    return analysis.load(DATA)


def test_dataset_shape(df):
    assert len(df) == 74
    assert df["location_id"].is_unique


def test_descriptives_match_source_workbook(df):
    desc = analysis.descriptives(df)
    assert desc.loc["annual_sales_k", "mean"] == pytest.approx(1059.38, abs=0.01)
    assert desc.loc["sqft", "mean"] == pytest.approx(2580.47, abs=0.01)


def test_iqr_outliers_flags_extremes():
    toy = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
    assert analysis.iqr_outliers(toy, "x")["x"].tolist() == [100]


def test_simple_regressions_cover_every_predictor(df):
    result = analysis.simple_regressions(df)
    assert set(result["predictor"]) == set(analysis.PREDICTORS)
    assert result["r_squared"].between(0, 1).all()


def test_missing_column_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"sqft": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        analysis.load(bad)


def test_run_writes_outputs(tmp_path):
    summary = analysis.run(DATA, tmp_path)
    assert summary["n_locations"] == 74
    assert (tmp_path / "figures" / "predictor_scatterplots.png").exists()
