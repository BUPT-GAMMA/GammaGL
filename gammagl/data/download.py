import os
import os.path as osp
import ssl
import sys
import urllib
from tqdm import tqdm
from typing import Optional

from .makedirs import makedirs


def download_url(url: str, folder: str, log: bool = True,
                 filename: Optional[str] = None):
    r"""Downloads the content of an URL to a specific folder.

        Parameters
        ----------
        url: str
            The url.
        folder: str
            The folder.
        log: bool, optional
            If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
        filename: str, optional
            The name of the file.

    """

    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]
    if os.environ.get('GGL_GITHUB_PROXY') == 'TRUE' and ('raw.githubusercontent.com' in url or 'github.com' in url):
        url = 'https://ghproxy.com/' + url
    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log:
        print(f'Downloading {url}', file=sys.stderr)

    makedirs(folder)

    context = ssl._create_unverified_context()
    response = urllib.request.urlopen(url, context=context)

    file_size = response.getheader('Content-Length', '0')

    # print(f"downloading {filename} ...")
    file_size = int(file_size)
    if file_size == 0:
        print(f"Remote file size not found.")

    with open(path, 'wb') as f:
        # workaround for https://bugs.python.org/issue42853
        # add download progress bar
        with tqdm(total=file_size, unit='B', unit_divisor=1024, unit_scale=True, desc=f'{filename}') as pbar:
            chunk_size = 10 * 1024 * 1024
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(chunk_size)

    return path
