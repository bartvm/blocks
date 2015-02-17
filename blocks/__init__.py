"""The blocks library for parametrized Theano ops."""
import os.path
from pkg_resources import get_distribution, DistributionNotFound

from blocks.config_parser import config  # flake8: noqa


try:
    DIST = get_distribution('blocks')
    DIST_LOC = os.path.normcase(DIST.location)
    HERE = os.path.normcase(__file__)
    if not HERE.startswith(os.path.join(DIST_LOC, 'blocks')):
        raise DistributionNotFound
except DistributionNotFound:
    __version__ = 'not installed'
else:
    __version__ = DIST.version
