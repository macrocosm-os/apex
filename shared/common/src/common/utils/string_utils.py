def escape_like_pattern(s: str) -> str:
    """
    Escape special characters in a string for use in SQL LIKE/ILIKE patterns.
    This prevents LIKE pattern injection where user input containing % or _
    could match unintended patterns.
    """
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
