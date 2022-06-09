from .amazon import Amazon
from .coauthor import Coauthor
from .tu_dataset import TUDataset
from .planetoid import Planetoid
from .reddit import Reddit
from .imdb import IMDB
from .entities import Entities
from .flickr import Flickr
from .wikipedia_network import WikipediaNetwork
from .webkb import WebKB
from .ppi import PPI

__all__ = [
    'Amazon',
    'Coauthor',
    'TUDataset',
    'Planetoid',
    'Reddit',
    'IMDB',
    'Entities',
    'Flickr',
    'WikipediaNetwork',
    'WebKB',
    'PPI'
]

classes = __all__
