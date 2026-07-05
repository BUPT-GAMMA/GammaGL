import os
import tempfile
from pathlib import Path
from gammagl.utils.data_utils import download_file,save_cache,load_cache


def test_download_file(tmp_path):
    source_path = tmp_path / "source.txt"
    source_path.write_text("GammaGL", encoding="utf-8")
    save_path = tmp_path / "downloaded.txt"
    url = source_path.as_uri()
    downloaded_path = download_file(save_path, url)
    assert os.path.exists(downloaded_path), "File was not downloaded correctly."
    assert Path(downloaded_path).read_text(encoding="utf-8") == "GammaGL"

def test_save_load_cache():
    test_obj = {"key1": "value1", "key2": 2}
    cache_path = tempfile.mktemp()
    save_cache(test_obj, cache_path)
    assert os.path.exists(cache_path), "Cache file was not created."
    loaded_obj = load_cache(cache_path)
    assert loaded_obj == test_obj, "Loaded object does not match the saved object."
