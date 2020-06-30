from functools import wraps
import time

from termcolor import colored
from colorama import init

init()


def pretty(func):
    """Prints some pretty info arround an algorithm call."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        fname = colored(func.__name__.replace("_", " "), "green", attrs=["bold"])
        print(f"Running {fname}... ", flush=True, end="")
        s = time.perf_counter()
        res = func(*args, **kwargs)
        elapsed = time.perf_counter() - s
        print("done! ✨ 🎉 ✨\n")
        print(f"{colored('Result', attrs=['bold'])}: \n{res}")
        elapsed = colored(f"{elapsed:0.3f}", "yellow", attrs=["bold"])
        print(f"\nExecuted in {elapsed} seconds.")
        return res

    return wrapped
