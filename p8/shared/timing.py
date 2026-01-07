import time
from contextlib import contextmanager

@contextmanager
def timed(name: str):
    start = time.time()
    yield
    end = time.time()
    print(f"[TIMING] {name}: {end - start:.2f}s")
