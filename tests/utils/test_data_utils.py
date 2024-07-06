import os
import tempfile
from gammagl.utils.data_utils import download_file,save_cache,load_cache
def test_download_file():
    url = "https://raw.githubusercontent.com/tensorflow/tensorflow/master/README.md"
    save_path = tempfile.mktemp()
    downloaded_path = download_file(save_path, url)
    assert os.path.exists(downloaded_path), "File was not downloaded correctly."

def test_save_load_cache():
    test_obj = {"key1": "value1", "key2": 2}
    cache_path = tempfile.mktemp()
    save_cache(test_obj, cache_path)
    assert os.path.exists(cache_path), "Cache file was not created."
    loaded_obj = load_cache(cache_path)
    assert loaded_obj == test_obj, "Loaded object does not match the saved object."

def run_tests():
    test_download_file()
    print("test_download_file passed.")   
    test_save_load_cache()
    print("test_save_load_cache passed.")

if __name__ == "__main__":
    run_tests()
