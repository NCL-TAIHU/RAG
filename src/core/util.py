
def coalesce(*args):
    """
    Returns the first non-None argument from the provided arguments.
    
    Args:
        *args: A variable number of arguments to check.
        
    Returns:
        The first non-None argument, or None if all are None.
    """
    for arg in args:
        if arg is not None:
            return arg
    return None

def get(ls: list, index: int, default=None):
    """
    Returns the element at the specified index from the list, or a default value if the index is out of bounds.
    
    Args:
        ls (list): The list to retrieve the element from.
        index (int): The index of the element to retrieve.
        default: The value to return if the index is out of bounds.
        
    Returns:
        The element at the specified index, or the default value if the index is out of bounds.
    """
    try:
        return ls[index]
    except IndexError:
        return default