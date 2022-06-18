from .amazon import Amazon
from .coauthor import Coauthor
from .tu_dataset import TUDataset
from .planetoid import Planetoid
from .reddit import Reddit
from .imdb import IMDB
from .entities import Entities
from .flickr import Flickr
from .hgb import HGBDataset
from .wikipedia_network import WikipediaNetwork
from .webkb import WebKB


__all__ = [
    'Amazon',
    'Coauthor',
    'TUDataset',
    'Planetoid',
    'Reddit',
    'IMDB',
    'Entities',
    'Flickr',
    'HGBDataset'
    'WikipediaNetwork',
    'WebKB'

]

classes = __all__
