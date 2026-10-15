"""Helpers shared by the converters: the error type and the precondition check (a converter refuses what it cannot convert faithfully)."""


class ConversionError(Exception):
    """The checkpoint cannot be converted faithfully."""


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ConversionError(message)
