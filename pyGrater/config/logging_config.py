#%%
import logging
from pathlib import Path
from datetime import datetime

class CustomFormatter(logging.Formatter):

    cyan = "\x1b[96;20m"
    green = "\x1b[92;20m"
    yellow = "\x1b[93;20m"
    red = "\x1b[91;20m"
    bold_red = "\x1b[91;1m"
    reset = "\x1b[0m"
    format = "%(levelname)s - %(message)s (%(filename)s:line %(lineno)d)"

    
    FORMATS = {
        logging.DEBUG: cyan + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    
def setup_logger(name='pyGrater', level=logging.DEBUG, log_to_file=False, log_dir=None):
    """
    Configure and return a logger with consistent formatting.

    Parameters
    ----------
    name : str
        Logger name (default: 'pyGrater')
    level : int
        Logging level (default: DEBUG)
    log_to_file : bool
        Whether to also log to file (default: False)
    log_dir : str or Path, optional
        Directory for log files (default: ./logs/)
    """
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    #Prevent duplicate handlers
    if logger.handlers:
        return logger

    # create console handler with colored output
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    
    # ==========================================================
    # OPTIONAL FILE LOGGING (NON-COLORED)
    # ==========================================================
    if log_to_file:
        if log_dir is None:
            log_dir = Path.cwd() / "logs"
        else:
            log_dir = Path(log_dir)

        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pyGrater_{timestamp}.log"

        # File formatter (no colors)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(module)s - %(message)s (%(filename)s:%(lineno)d)",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # File gets ALL messages
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")
    
    return logger

if __name__ == "__main__":
    # Example usage
    logger = setup_logger(log_to_file=False)
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
# %%
