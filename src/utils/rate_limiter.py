import time
from threading import Lock

class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = Lock()

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self.lock:
                now = time.time()
                
                # Remove calls older than the period
                self.calls = [call for call in self.calls if now - call < self.period]
                
                if len(self.calls) >= self.max_calls:
                    sleep_time = self.period - (now - self.calls[0])
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                result = func(*args, **kwargs)
                self.calls.append(time.time())
                return result
        return wrapper

# Usage:
# @RateLimiter(max_calls=30, period=60)
# def your_function():
#     ...