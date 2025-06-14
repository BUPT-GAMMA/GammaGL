from .amazon import Amazon
from .coauthor import Coauthor
from .ngsim import NGSIM_US_101
from .tu_dataset import TUDataset
from .planetoid import Planetoid
from .reddit import Reddit
from .ppi import PPI
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
from .aminer import AMiner
from .polblogs import PolBlogs
from .wikics import WikiCS
from .blogcatalog import BlogCatalog
from .molecule_net import MoleculeNet
from .facebook import FacebookPagePage
from .acm4heco import ACM4HeCo
from .yelp import Yelp
from .bail import Bail
from .credit import Credit
from .acm4dhn import ACM4DHN
from .acm4rohe import ACM4Rohe
from .actor import Actor
from .deezer_europe import DeezerEurope
from .ADDataset import ADDataset
from .pcqm4mv2 import (
    PCQM4Mv2,
    PCQM4Mv2Dataset,
    PCQM4Mv2GraphDataset,
    PCQM4Mv2SVDGraphDataset,
    PCQM4Mv2StructuralGraphDataset
)

__all__ = [
    'ACM4HeCo',
    'Amazon',
    'Coauthor',
    'TUDataset',
    'Planetoid',
    'PPI',
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
    'ZINC',
    'AMiner',
    'PolBlogs',
    'WikiCS',
    'MoleculeNet',
    'FacebookPagePage',
    'NGSIM_US_101',
    'Yelp',
    'Bail',
    'Credit',
    'ACM4DHN',
    'ACM4Rohe',
    'Actor',
    'DeezerEurope',
    'ADDataset',
    'PCQM4Mv2',
    'PCQM4Mv2Dataset',
    'PCQM4Mv2GraphDataset',
    'PCQM4Mv2SVDGraphDataset',
    'PCQM4Mv2StructuralGraphDataset',
]

classes = __all__
