import ssl
import sys
import urllib.request
import os.path as osp
from .makedirs import makedirs


def download_url(url: str, folder: str, log: bool = True):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    #filename = url.rpartition('/')[2]
    #filename = filename if filename[0] == '?' else filename.split('?')[0]
    #TODO:提出filename作为download_url函数的参数，因为各个数据集的url的格式并不完全一种方式能够处理
    filename = 'DBLP.zip'
    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log:
        print(f'Downloading {url}', file=sys.stderr)

    makedirs(folder)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        f.write(data.read())

    return path