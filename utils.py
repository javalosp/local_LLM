import logging
import sys

# Get a logger for this module. It inherits the root configuration.
logger = logging.getLogger(__name__)

class StreamToLogger:
    """
    A helper class to redirect a stream (like stdout or stderr) to a logger.
    This is required to write outputs from cpp libraries (e.g. LlamaCpp) to the 
    log file
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


def setup_logging(console_level=logging.WARNING, file_level=logging.INFO):
    """
    Manually configures the logger with separate levels for console and file.
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the lowest possible level on the logger itself

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] - %(name)s - %(message)s"
    )

    # --- Console Handler ---
    # Only shows WARNING and above on the screen
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(console_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # --- File Handler ---
    # Writes INFO and above to the log file
    file_handler = logging.FileHandler("local_llm.log", mode='w')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def setup_logging_(level=logging.INFO):
    """
    Configures the root logger for the application with a specified level.

    This should be called once at the start of the main script. It sets up
    logging to output to both a file (`local_llm.log`) and the console.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] - %(name)s - %(message)s",
        handlers=[
            #logging.FileHandler("local_llm.log"), # Append log info to existing file
            logging.FileHandler("local_llm.log", mode='w'), # Overwrite log file
            logging.StreamHandler(sys.stdout)
        ]
    )

def initialise_logging(verbosity):
    """
    Initialises the application's logging based on a verbosity level.

    Args:
        verbosity (int): The verbosity level (0, 1, or 2).
        Defaults to 1 (Normal)
    """
    log_levels = {
        0: "Silent (Log Warnings, Console Critical)",
        1: "Normal (Log Info, Console Warnings)",
        2: "Debug (Log Debug, Console Info)",
    }

    if verbosity == 0:
        setup_logging(console_level=logging.CRITICAL, file_level=logging.WARNING)
    elif verbosity == 1:
        setup_logging(console_level=logging.WARNING, file_level=logging.INFO)
    elif verbosity == 2:
        setup_logging(console_level=logging.INFO, file_level=logging.DEBUG)
    
    logger = logging.getLogger(__name__)
    logger.info("Logging initialised.")
    logger.info(f"Verbosity level set to {verbosity}: {log_levels.get(verbosity, 'Unknown')}")


def get_query_from_file(file_path):
    """
    Reads a query from the specified text file. It's useful for running complex
    or predefined queries without hardcoding them into the main script.

    The function handles cases where the file does not exist by printing an
    error message to the console and returning None.

    Args:
        file_path (str): The full path to text file containing the query.

    Returns:
        str or None: The query text read from the file with leading and
        trailing whitespace removed, or None if the file cannot be found.
    """

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            logger.info(f"Successfully read query from {file_path}")
            return f.read().strip()
    except FileNotFoundError:
        logger.exception(f"Error: Query file not found at {file_path}")
        return None