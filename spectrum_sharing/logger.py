""" logger.py. 

Logger definition. """

import logging
import colorlog
import os

logger = logging.getLogger(__name__) # __name__ makes logger specific to each module
logger.setLevel(logging.INFO)  # Set the overall logging level for the logger
logger.propagate = False  # Prevent the log messages from propagating to the root logger        

if not logger.handlers:
    # Add file handler
    log_dir = str(os.getcwd()) + "/logging"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(f"{log_dir}/app.log", encoding="utf-8", mode="a")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        fmt="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M"
    )
    file_handler.setFormatter(file_formatter)

    # Add stream handler with color
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    color_formatter = colorlog.ColoredFormatter(
        "{log_color}{asctime} - {levelname} - {message}",
        datefmt="%Y-%m-%d %H:%M",
        style="{",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
    stream_handler.setFormatter(color_formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)