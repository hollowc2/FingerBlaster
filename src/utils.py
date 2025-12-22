"""Error handling utilities."""

import logging
from functools import wraps
from typing import Callable

logger = logging.getLogger("FingerBlaster")


def handle_ui_errors(func: Callable) -> Callable:
    """Decorator for handling UI-related errors."""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except (AttributeError, KeyError) as e:
            logger.debug(f"UI error in {func.__name__}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
    return wrapper


def handle_sync_errors(func: Callable) -> Callable:
    """Decorator for handling synchronous errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (AttributeError, KeyError, ValueError, TypeError) as e:
            logger.debug(f"Error in {func.__name__}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            return None
    return wrapper
