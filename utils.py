from config import Config
import sys
import tqdm

def eprint(*args, **kwargs):
    return print(*args, file=sys.stderr, **kwargs)

def tprint(*args, **kwargs):
    return tqdm.write(*args, file=sys.stderr, **kwargs)

