import logging
import sys

# Get a logger for this module. It inherits the root configuration.
logger = logging.getLogger(__name__)

def setup_logging():
    """
    Configures the root logger for the application.

    This should be called once at the start of the main script. It sets up
    logging to output to both a file (`local_llm.log`) and the console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler("local_llm.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )


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