import os
import re
from typing import NamedTuple

import hvac
import pandas as pd
import s3fs

from src.logger import get_logger
from src.variables import VAULT_ENDPOINT_URL, VAULT_PATH

logger = get_logger(__name__)


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
        int(p) if p.isdigit() else p.lower().replace("_", " ").strip() for p in parts
    ]


def get_s3_credentials(
    vault_endpoint_url: str = VAULT_ENDPOINT_URL, vault_path: str = VAULT_PATH
) -> dict[str | None, str | None]:
    """Retrieve S3 credentials from Vault.

    Authenticates using the VAULT_TOKEN environment variable and reads the
    secret at the given path from the onyxia-kv mount point.

    Parameters
    ----------
    vault_endpoint_url : str
        URL of the Vault instance.
    vault_path : str
        Path to the secret containing AWS credentials.

    Returns
    -------
    tuple[str | None, str | None]
        AWS access key ID and secret access key, or (None, None) on failure.
    """
    try:
        client = hvac.Client(url=vault_endpoint_url, token=os.getenv("VAULT_TOKEN"))

        secret = client.secrets.kv.read_secret_version(
            path=vault_path, mount_point="onyxia-kv"
        )
    except Exception as e:
        logger.warning("vault_credentials_failed", extra={"error": str(e)})
        return None, None

    return secret["data"]["AWS_ACCESS_KEY_ID"], secret["data"]["AWS_SECRET_ACCESS_KEY"]


def get_fs(s3_endpoint_url: str | None) -> s3fs.S3FileSystem:
    """Build an S3FileSystem, authenticated if Vault credentials are available.

    Falls back to anonymous access if credentials cannot be retrieved.

    Parameters
    ----------
    s3_endpoint_url : str or None
        Custom S3 endpoint URL.

    Returns
    -------
    s3fs.S3FileSystem
    """
    aws_access_key_id, aws_secret_access_key = get_s3_credentials()
    if aws_access_key_id is None or aws_secret_access_key is None:
        return s3fs.S3FileSystem(
            anon=True, client_kwargs={"endpoint_url": s3_endpoint_url}
        )
    return s3fs.S3FileSystem(
        key=aws_access_key_id,
        secret=aws_secret_access_key,
        client_kwargs={"endpoint_url": s3_endpoint_url},
    )


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
        Custom endpoint URL.

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
        Custom endpoint URL.

    Returns
    -------
    pd.DataFrame
        Parsed contents of the CSV file.
    """
    fs = get_fs(endpoint_url)
    with fs.open(f"{bucket}/{key}") as f:
        return pd.read_csv(f)


def detect_datetime_candidates(df: pd.DataFrame) -> list[str]:
    """Return column names that are likely datetime columns.

    A column is considered a candidate if it already has a datetime dtype, or
    if more than 90% of its string values parse successfully as dates (trying
    both day-first and month-first formats).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to inspect.

    Returns
    -------
    list[str]
        Names of columns identified as datetime candidates.
    """
    return [
        col
        for col in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[col])
        or (
            df[col].dtype == object
            and max(
                pd.to_datetime(df[col], errors="coerce", dayfirst=False).notna().mean(),
                pd.to_datetime(df[col], errors="coerce", dayfirst=True).notna().mean(),
            )
            > 0.9
        )
    ]


def set_datetime_index(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Set a column as a sorted datetime index.

    Tries month-first parsing first, falls back to day-first if more than 10%
    of values fail to parse.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    time_col : str
        Name of the column to use as the index.

    Returns
    -------
    pd.DataFrame
        New DataFrame with ``time_col`` as a sorted DatetimeIndex.
    """
    parsed = pd.to_datetime(df[time_col], errors="coerce", dayfirst=False)
    if parsed.isna().mean() > 0.1:
        parsed = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
    df = df.copy()
    df[time_col] = parsed
    return df.set_index(time_col).sort_index()
