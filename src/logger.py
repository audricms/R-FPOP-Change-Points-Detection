import logging
import sys

from pythonjsonlogger import jsonlogger


def get_logger(name: str) -> logging.Logger:
    """Return a JSON-formatted stdout logger.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        Configured logger that emits one JSON object per line to stdout.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            jsonlogger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
