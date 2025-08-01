import contextlib
import logging
import os
import sys
from pathlib import Path

from tempo.utils.common import get_date_str, get_dir_path

log_level_map = {
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "OFF": logging.CRITICAL + 1,
}

log_level = os.environ.get("Tempo_LOG_LEVEL", "INFO").upper().strip()
if log_level not in log_level_map:
    raise Exception(f"Unknown log level requested {log_level}")
log_level = log_level_map[log_level]

log_to_file = os.environ.get("TEMPO_LOG_TO_FILE", "FALSE").upper().strip() == "TRUE"
log_datefmt = "%Y-%m-%d,%H:%M:%S"
log_fmt = (
    "%(asctime)s,%(msecs)03d %(levelname)-8s [%(process)d,%(filename)s:%(lineno)d]: %(message)s"
)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Clear existing handlers
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    # Stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(logging.Formatter(log_fmt, log_datefmt))
    logger.addHandler(stream_handler)

    # File handler (if required)
    if log_to_file:
        logs_dir = Path(get_dir_path()) / "logs"
        with contextlib.suppress(FileExistsError):
            logs_dir.mkdir(parents=True, exist_ok=True)

        date_str = get_date_str()
        file_handler = logging.FileHandler(logs_dir / f"{date_str}.log")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_fmt, log_datefmt))
        logger.addHandler(file_handler)

    # Prevent log messages from being propagated to the root logger
    logger.propagate = False

    return logger
