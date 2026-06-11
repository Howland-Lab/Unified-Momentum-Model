from pathlib import Path
import functools
import polars as pl

def cache_polars(cache_file):
    """
    Decorator function for caching Polars DataFrame using CSV format.

    Parameters:
    - cache_file (Union[str, Path]): The path to the cache file.

    Returns:
    - Callable: Decorator function to be applied to another function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_filepath = Path(cache_file)
            cache_filepath.parent.mkdir(exist_ok=True, parents=True)
            regenerate = kwargs.pop("regenerate", False)

            if not regenerate and cache_filepath.exists():
                return pl.read_csv(cache_filepath)
            else:
                df = func(*args, **kwargs)
                df.write_csv(cache_filepath)
                return df

        return wrapper
    return decorator