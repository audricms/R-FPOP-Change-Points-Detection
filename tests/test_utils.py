import pandas as pd

from src.utils import detect_datetime_candidates, natural_key, set_datetime_index


def test_natural_key_sorts_numerically():
    files = [
        "example_time_series_10.csv",
        "example_time_series_2.csv",
        "example_time_series_1.csv",
        "example_time_series_3.csv",
    ]
    assert sorted(files, key=natural_key) == [
        "example_time_series_1.csv",
        "example_time_series_2.csv",
        "example_time_series_3.csv",
        "example_time_series_10.csv",
    ]


def test_detect_datetime_candidates_iso_format():
    df = pd.DataFrame(
        {"date": ["2020-01-01", "2020-01-02", "2020-01-03"], "value": [1, 2, 3]}
    )
    assert detect_datetime_candidates(df) == ["date"]


def test_detect_datetime_candidates_dayfirst_format():
    df = pd.DataFrame(
        {"date": [f"{d:02d}/01/2020" for d in range(1, 20)], "value": range(19)}
    )
    assert detect_datetime_candidates(df) == ["date"]


def test_detect_datetime_candidates_no_datetime():
    df = pd.DataFrame({"name": ["alice", "bob"], "value": [1, 2]})
    assert detect_datetime_candidates(df) == []


def test_set_datetime_index_sets_index_and_sorts():
    df = pd.DataFrame(
        {"date": ["2020-01-03", "2020-01-01", "2020-01-02"], "value": [3, 1, 2]}
    )
    result = set_datetime_index(df, "date")
    assert result.index.name == "date"
    assert pd.api.types.is_datetime64_any_dtype(result.index)
    assert list(result.index) == sorted(result.index)


def test_set_datetime_index_dayfirst_format():
    df = pd.DataFrame(
        {"date": [f"{d:02d}/01/2020" for d in range(1, 20)], "value": range(19)}
    )
    result = set_datetime_index(df, "date")
    assert result.index.isna().sum() == 0
