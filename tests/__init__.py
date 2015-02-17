import logging
import sys
from functools import wraps

from six import StringIO

import blocks


def silence_printing(test):
    @wraps(test)
    def wrapper(*args, **kwargs):
        stdout = sys.stdout
        sys.stdout = StringIO()
        logger = logging.getLogger(blocks.__name__)
        old_level = logger.level
        logger.setLevel(logging.ERROR)
        try:
            test(*args, **kwargs)
        finally:
            sys.stdout = stdout
            logger.setLevel(old_level)
    return wrapper
