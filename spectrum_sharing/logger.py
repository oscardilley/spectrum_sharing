""" logger.py. 

Logger definition. """


import logging
import colorlog
import os
import sys
import os.path as path

# Get the main script name (the one run directly)
def get_main_script_name():
    main_path = sys.argv[0]
    if main_path:
        return os.path.splitext(os.path.basename(main_path))[0]
    return "app"

main_script_name = get_main_script_name()

logger = logging.getLogger(main_script_name)
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    # Create logging directory if it doesn't exist
    log_dir = os.path.join(os.getcwd(), "logging")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{main_script_name}.log")

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        fmt="{asctime} - {levelname} - {name} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M"
    )
    file_handler.setFormatter(file_formatter)

    # Stream handler with color
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

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)