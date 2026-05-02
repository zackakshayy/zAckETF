import os

def cpu_count_sensible() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count() or 2
    return max(2, n - 1)

def joblib_n_jobs() -> int:
    n = cpu_count_sensible()
    return max(1, n - 1)
