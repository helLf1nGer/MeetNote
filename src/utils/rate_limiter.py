import functools
import logging
import time
from threading import Lock

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Sliding-window rate limiter usable as a decorator.

    The lock guards only the bookkeeping, never the wrapped call or the sleep.
    Holding it across either would serialize every caller and turn this into a
    mutex that happens to rate limit, rather than a rate limiter.
    """

    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = Lock()

    def acquire(self):
        """Block until a call slot is available, then reserve it."""
        while True:
            with self.lock:
                now = time.monotonic()
                self.calls = [call for call in self.calls if now - call < self.period]

                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return

                # Wait for the oldest call to fall out of the window.
                sleep_time = self.period - (now - self.calls[0])

            if sleep_time > 0:
                logger.info("[RateLimiter] Limit reached; waiting %.2fs.", sleep_time)
                time.sleep(sleep_time)

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.acquire()
            return func(*args, **kwargs)
        return wrapper


# Usage:
# @RateLimiter(max_calls=30, period=60)
# def your_function():
#     ...
