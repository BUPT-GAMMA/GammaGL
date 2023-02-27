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
from .modelnet40 import ModelNet40
from .dblp import DBLP
from .ca_grqc import CA_GrQc
from .zinc import ZINC

__all__ = [
    'Amazon',
    'Coauthor',
    'TUDataset',
    'Planetoid',
    'Reddit',
    'IMDB',
    'Entities',
    'Flickr',
    'HGBDataset',
    'WikipediaNetwork',
    'WebKB',
    'ModelNet40',
    'DBLP',
    'CA_GrQc',
    'ZINC'
]

classes = __all__
