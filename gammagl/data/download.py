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

    if osp.exists(path) and osp.getsize(path) > 0:  # pragma: no cover
        # Check if existing file size matches expected
        context = ssl._create_unverified_context()
        try:
            req = urllib.request.Request(url, method='HEAD')
            response = urllib.request.urlopen(req, context=context)
            expected_size = int(response.getheader('Content-Length', '0'))
            if expected_size > 0 and osp.getsize(path) == expected_size:
                if log:
                    print(f'Using existing file {filename}', file=sys.stderr)
                return path
            else:
                if log:
                    print(f'Existing file {filename} is incomplete, re-downloading...', file=sys.stderr)
                os.remove(path)
        except Exception:
            if log:
                print(f'Re-downloading {filename}...', file=sys.stderr)
            os.remove(path)

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

    downloaded = 0
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
                chunk_len = len(chunk)
                downloaded += chunk_len
                pbar.update(chunk_len)

    # Verify downloaded file size
    if file_size > 0 and downloaded < file_size:
        if log:
            print(f'Download incomplete ({downloaded}/{file_size} bytes), removing partial file.', file=sys.stderr)
        os.remove(path)
        raise RuntimeError(f'Failed to download {url}: got {downloaded}/{file_size} bytes')

    return path


def download_google_url(id: str, folder: str,
                        filename: str, log: bool = True):
    r"""Downloads the content of a Google Drive ID to a specific folder."""
    url = f'https://drive.usercontent.google.com/download?id={id}&confirm=t'
    return download_url(url, folder, log, filename)
