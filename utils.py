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
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Query file not found at {file_path}")
        return None