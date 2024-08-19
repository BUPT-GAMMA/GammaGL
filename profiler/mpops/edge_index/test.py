from gammagl.data.download import download_url


url = "https://github.com/gyzhou2000/gammagl_files/raw/main/"
files = ['cora.npy', 'pubmed.npy', 'ogbn-arxiv.npy']
for file in files:
    download_url(url + file, "./")
