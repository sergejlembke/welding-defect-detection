import logging
from pathlib import Path


def setup_logger(log_file: Path, logger_name: str = "pipeline_logger") -> logging.Logger:
    """Configure file and console logging handlers for model pipelines.

    Args:
        log_file (Path): Target file path for writing log output.
        logger_name (str): Unique name identifier for the logger instance.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File log handler
    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console stream log handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger