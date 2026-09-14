import sys
from .backends import worker

if __name__ == '__main__':
    worker(sys.argv[1])
