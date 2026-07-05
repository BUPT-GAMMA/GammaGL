# coding=utf-8
import os
import pickle
import urllib.request



def download_file(path, url_or_urls):
    path = os.fspath(path)
    if not isinstance(url_or_urls, list):
        urls = [url_or_urls]
    else:
        urls = url_or_urls

    last_except = None
    for url in urls:
        try:
            download_path = path
            if os.path.isdir(download_path):
                download_path = os.path.join(download_path, os.path.basename(url))
            os.makedirs(os.path.dirname(os.path.abspath(download_path)), exist_ok=True)
            urllib.request.urlretrieve(url, download_path)
            return download_path
        except Exception as e:
            last_except = e
            print(e)

    raise last_except


def save_cache(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)


def load_cache(path):
    # if not os.path.exists(path):
    #     return None

    with open(path, "rb") as f:
        return pickle.load(f)
