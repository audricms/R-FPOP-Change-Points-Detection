import re
import xml.etree.ElementTree as ET
from io import StringIO
from urllib.parse import quote, urlsplit, urlunsplit

import pandas as pd
import requests


def natural_key(s: str):
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
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def encode_url_path(url: str) -> str:
    """Return a URL with its path percent-encoded while preserving scheme, host and query.

    Parameters
    ----------
    url : str
        Raw URL whose path may contain special characters.

    Returns
    -------
    str
        URL with the path component percent-encoded.
    """
    parts = urlsplit(url)
    encoded_path = quote(parts.path, safe="/")
    return urlunsplit(
        (parts.scheme, parts.netloc, encoded_path, parts.query, parts.fragment)
    )


def build_public_toy_csv_url(base_url: str, filename: str) -> str:
    """Build a public URL for a CSV file stored in S3.

    Parameters
    ----------
    base_url : str
        Base URL of the S3 bucket and prefix.
    filename : str
        Name of the CSV file (without leading slash).

    Returns
    -------
    str
        Fully encoded URL pointing to the file.
    """
    normalized_base = encode_url_path(base_url.rstrip("/"))
    return f"{normalized_base}/{quote(filename)}"


def list_s3_csv_files(base_url: str) -> list[str]:
    """List CSV filenames available under a public S3 prefix.

    Queries the S3 ListObjectsV2 XML API and returns only the filenames
    (without the prefix path) of objects whose key ends with ``.csv``.

    Parameters
    ----------
    base_url : str
        URL containing the scheme, host, bucket and optional prefix

    Returns
    -------
    list[str]
        Filenames sorted with natural ordering.

    Raises
    ------
    requests.HTTPError
        If the listing request returns a non-2xx status code.
    """
    parts = urlsplit(base_url)
    path_parts = parts.path.lstrip("/").split("/", 1)
    bucket = path_parts[0]
    prefix = (path_parts[1].rstrip("/") + "/") if len(path_parts) > 1 else ""
    list_url = (
        f"{parts.scheme}://{parts.netloc}/{bucket}?list-type=2&prefix={quote(prefix)}"
    )
    response = requests.get(list_url, timeout=10)
    response.raise_for_status()
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    root = ET.fromstring(response.text)
    return sorted(
        [
            key.text.split("/")[-1]
            for key in root.findall(".//s3:Key", ns)
            if key.text and key.text.endswith(".csv")
        ],
        key=natural_key,
    )


def read_csv_from_public_url(url: str) -> pd.DataFrame:
    """Read a CSV file from a public HTTP URL into a DataFrame.

    Parameters
    ----------
    url : str
        Fully encoded public URL of the CSV file.

    Returns
    -------
    pd.DataFrame
        Parsed contents of the CSV file.

    Raises
    ------
    requests.HTTPError
        If the request returns a non-2xx status code.
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))
