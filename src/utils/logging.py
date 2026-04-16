import logging
import sys


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(stream=sys.stdout, level=level, force=True)
    logger = logging.getLogger("curriculum_learning")
    logger.setLevel(level)
    for noisy in ("httpx", "urllib3", "requests", "py4j"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logger
