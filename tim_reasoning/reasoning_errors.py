from enum import Enum


class ReasoningErrors(Enum):
    """Types of errors for reasoning"""

    # Define the different error types.
    INVALID_STATE = (
        "Invalid state.",  # title
        "The state provided is not valid.",  # description
    )
    MISSING_PREVIOUS = (
        "Previous step incomplete.",  # title
        "Some of the previous states are incomplete.",  # description
    )
    NOT_STARTED = ("No completed nodes", "The task graph hasn't started, there were no tasks recorded.")
    UNEXPECTED_ERROR = (
        "An unexpected error occurred.",
        "Please contact support for assistance.",
    )
