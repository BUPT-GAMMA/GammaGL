from .amazon import Amazon
from .coauthor import Coauthor
from .tu_dataset import TUDataset
from .planetoid import Planetoid
from .reddit import Reddit
from .imdb import IMDB
from .entities import Entities
from .flickr import Flickr
from .dblp import HGBDataset

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
]

classes = __all__