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
<<<<<<< HEAD
from .dblp import DBLP

=======
from .ca_grqc import CA_GrQc
from .zinc import ZINC
>>>>>>> 7468ee0fb1408ba047a7f0dce289b33e88297c41

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
<<<<<<< HEAD
    'DBLP'

=======
    'CA_GrQc',
    'ZINC'
>>>>>>> 7468ee0fb1408ba047a7f0dce289b33e88297c41
]

classes = __all__
