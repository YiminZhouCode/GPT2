def save_to_file(content, filepath):
    """
    Save the given content to a file.

    Args:
        content (str): The content to be saved.
        filepath (str): The path to the file where the content will be saved.
    """
    with open(filepath, 'w') as file:
        file.write(content)
