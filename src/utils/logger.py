import logging
import os
from threading import Lock

# A global dictionary to store loggers and a lock for thread-safe operations
_loggers = {}
_logger_lock = Lock()


def get_logger(
    module_name: str, client_id: int, log_dir: str = "log"
) -> logging.Logger:
    global _loggers

    # Using thread-safe lock to prevent race conditions
    with _logger_lock:
        if module_name in _loggers:
            return _loggers[module_name]

        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{client_id}.log")

        # Create a new logger instance
        logger = logging.getLogger(module_name)
        logger.setLevel(logging.INFO)

        # Create a file handler
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - Line %(lineno)d - %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(handler)

        # Prevent propagation to the root logger
        logger.propagate = False

        _loggers[module_name] = logger
        return logger
