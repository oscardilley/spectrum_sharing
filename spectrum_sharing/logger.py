""" logger.py. 

Logger definition. """

import logging
import colorlog
import os

logger = logging.getLogger("Logger")
logger.setLevel(logging.INFO)  # Set the overall logging level for the logger

# File handler to log messages to a file (INFO and above)
file_handler = logging.FileHandler(str(os.getcwd()) + "/logging/app.log", encoding="utf-8", mode="a")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(
    fmt="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M"
)
file_handler.setFormatter(file_formatter)

# Stream handler to log messages to the console (INFO and above, with color)
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