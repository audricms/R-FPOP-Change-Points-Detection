import re
from typing import NamedTuple

import pandas as pd
import s3fs


class QuadPiece(NamedTuple):
    """A named tuple representing a piece of a piecewise quadratic function.

    Fields
    ------
    a: float
        Left bound of the interval (open)
    b: float
        Right bound of the interval (closed)
    A: float
        Quadratic coefficient (theta^2)
    B: float
        Linear coefficient (theta)
    C: float
        Constant term
    tau: int
        Index of the last changepoint associated with this piece
    """

    a: float
    b: float
    A: float
    B: float
    C: float
    tau: int


def natural_key(s: str) -> list[int | str]:
    """Split a string into a list of strings and integers for natural sorting.

    Parameters
    ----------
    s : str
        Input string to split.

    Returns
    -------
    list
        Alternating string and integer parts, suitable for use as a sort key.
    """
    parts = re.split(r"(\d+)", s)
    return [
        int(p) if p.isdigit() else p.lower().replace("_", "").strip() for p in parts
    ]


def get_fs(endpoint_url: str | None) -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": endpoint_url})


def list_s3_csv_files(
    bucket: str, prefix: str = "", endpoint_url: str | None = None
) -> list[str]:
    """List CSV filenames available under an S3 prefix.

    Parameters
    ----------
    bucket : str
        Name of the S3 bucket.
    prefix : str
        Key prefix to filter objects.
    endpoint_url : str, optional
        Custom endpoint URL (e.g. for MinIO).

    Returns
    -------
    list[str]
        Filenames sorted with natural ordering.
    """
    fs = get_fs(endpoint_url)
    path = f"{bucket}/{prefix.rstrip('/')}/" if prefix else bucket
    entries = fs.ls(path, detail=False)
    keys = [e.split("/")[-1] for e in entries if e.endswith(".csv")]
    return sorted(keys, key=natural_key)


def read_csv_from_s3(
    bucket: str, key: str, endpoint_url: str | None = None
) -> pd.DataFrame:
    """Read a CSV file from S3 into a DataFrame.

    Parameters
    ----------
    bucket : str
        Name of the S3 bucket.
    key : str
        Full object key of the CSV file.
    endpoint_url : str, optional
        Custom endpoint URL (e.g. for MinIO).

    Returns
    -------
    pd.DataFrame
        Parsed contents of the CSV file.
    """
    fs = get_fs(endpoint_url)
    with fs.open(f"{bucket}/{key}") as f:
        return pd.read_csv(f)
