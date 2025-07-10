from src.core.document import Document

def coalesce(*args):
    """
    Returns the first non-None argument from the provided arguments.
    
    Args:
        *args: A variable number of arguments to check.
        
    Returns:
        The first non-None argument, or None if all are None.
    """
    for arg in args:
        if callable(arg): result = arg()
        else: result = arg
        if result is not None:
            return result
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
    
def get_first_content(doc: Document) -> str:
        """
        Helper method to extract the first content from a Document.
        If the document has no content, returns an empty string.
        """
        if doc.content():
            first_field = next(iter(doc.content().values()))
            if first_field.contents:
                return first_field.contents[0]
            else: 
                #first field exists but is empty
                return ""
        return ""