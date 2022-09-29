''' Contains all available networks in v2i '''

# base class
from v2i.networks.base import v2iNetwork

# custom networks
from v2i.networks.merge import twoInMergeNetwork

__all__ = [
    'twoInMergeNetwork',
]
