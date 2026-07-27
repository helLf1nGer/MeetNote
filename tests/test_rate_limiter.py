"""Rate limiter behaviour under concurrency."""

import threading
import time

from utils.rate_limiter import RateLimiter


def test_allows_calls_up_to_the_limit_immediately():
    limiter = RateLimiter(max_calls=5, period=10)
    start = time.monotonic()
    for _ in range(5):
        limiter.acquire()
    assert time.monotonic() - start < 0.5


def test_blocks_once_the_window_is_full():
    limiter = RateLimiter(max_calls=2, period=1)
    limiter.acquire()
    limiter.acquire()

    start = time.monotonic()
    limiter.acquire()  # must wait for the window to roll over
    assert time.monotonic() - start >= 0.5


def test_window_slides():
    limiter = RateLimiter(max_calls=2, period=0.5)
    limiter.acquire()
    limiter.acquire()

    time.sleep(0.6)  # both calls fall out of the window

    start = time.monotonic()
    limiter.acquire()
    assert time.monotonic() - start < 0.2


def test_does_not_serialize_concurrent_calls():
    """
    Regression test.

    The limiter used to hold its lock across the wrapped call, so ten 0.2s
    requests ran back to back instead of together - a mutex that happened to
    rate limit. Under the limit, concurrent calls must overlap.
    """
    limiter = RateLimiter(max_calls=10, period=60)

    @limiter
    def slow_call():
        time.sleep(0.2)

    start = time.monotonic()
    threads = [threading.Thread(target=slow_call) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.monotonic() - start

    # Serialized would be ~2.0s; overlapping is ~0.2s.
    assert elapsed < 1.0


def test_decorator_returns_value_and_preserves_name():
    limiter = RateLimiter(max_calls=5, period=10)

    @limiter
    def add(a, b):
        """Adds."""
        return a + b

    assert add(2, 3) == 5
    assert add.__name__ == 'add'
    assert add.__doc__ == 'Adds.'
