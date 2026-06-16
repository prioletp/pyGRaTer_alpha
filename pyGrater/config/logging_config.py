#%%
import logging
import sys
import atexit
from pathlib import Path
from datetime import datetime


class TeeStream:
    """Writes to both the original stream and a log file.

    Used to tee sys.stdout so that every print() call also lands
    in the pyGrater log file, regardless of any verbose/talk flags.
    """

    def __init__(self, original_stream, log_file_handle):
        self._original = original_stream
        self._log = log_file_handle

    def write(self, text):
        self._original.write(text)
        self._log.write(text)
        self._log.flush()

    def flush(self):
        self._original.flush()
        self._log.flush()

    def isatty(self):
        return self._original.isatty()

    def fileno(self):
        return self._original.fileno()


def redirect_print_to_log(log_dir=None):
    """Redirect sys.stdout so every print() call also writes to a log file.

    Called automatically when pyGrater is imported.  You can call it
    manually to point at a different directory.

    Parameters
    ----------
    log_dir : str or Path, optional
        Directory for log files (default: ``./logs/`` relative to cwd).

    Returns
    -------
    Path or None
        Path to the active log file, or *None* if already redirected.
    """
    if isinstance(sys.stdout, TeeStream):
        return None  # already active – do not open a second file

    if log_dir is None:
        log_dir = Path.cwd() / "logs"
    else:
        log_dir = Path(log_dir)

    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pyGrater_{timestamp}.log"

    fh = open(log_file, "a", encoding="utf-8")
    sys.stdout = TeeStream(sys.stdout, fh)

    # Restore original stdout and close file on interpreter exit.
    atexit.register(lambda: setattr(sys, "stdout", sys.stdout._original))
    atexit.register(fh.close)

    return log_file

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
