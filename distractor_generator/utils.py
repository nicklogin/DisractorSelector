from xml.parsers.expat import model
import requests
import os

from zipfile import ZipFile


# Taken from https://stackoverflow.com/a/14260592
def download_url(url: str, save_path: str, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def download_word2vec_model():
    '''
    Downloads Word2Vec model used in this study
    '''
    print("Downloading Word2Vec model")
    model_dir = "gensim_models/skipgram_wikipedia_no_lemma"
    download_url(
        "http://vectors.nlpl.eu/repository/20/222.zip",
        "222.zip"
    )
    with open("222.zip", 'rb') as inp:
        file = ZipFile(inp)
        file.extractall(model_dir)
    os.remove("222.zip")
    print("Downloading Word2Vec model - complete")
