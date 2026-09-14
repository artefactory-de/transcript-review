"""Errors safe to show without including source passages."""


class AnonymizerError(Exception):
    """An actionable error containing no sensitive document values."""
