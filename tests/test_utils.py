from src.utils import natural_key


def test_natural_key_sorts_numerically():
    files = [
        "data example 10.csv",
        "data example2.csv",
        "data example 1.csv",
        "data example_3.csv",
    ]
    assert sorted(files, key=natural_key) == [
        "data example 1.csv",
        "data example2.csv",
        "data example_3.csv",
        "data example 10.csv",
    ]
