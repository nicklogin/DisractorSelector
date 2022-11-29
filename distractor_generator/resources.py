import pandas as pd

import os
import json
import shutil
import pickle

from collections import defaultdict
from ast import literal_eval
from zipfile import ZipFile

from gensim.models import KeyedVectors

from distractor_generator.utils import download_word2vec_model
from distractor_generator.bert_embedder import BertEmbedder


if not os.path.exists("sinonyms_compressed.pickle"):
    folder = "sinonyms_compressed"

    ## merge zip archives into one:
    with open("sinonyms_compressed.zip", "wb") as outp:
        for file in os.listdir(folder):
            if file.startswith("sinonyms_compressed.zip"):
                with open(os.path.join(folder, file), "rb") as inp:
                    shutil.copyfileobj(inp, outp)

    ## unzip and delete archive:
    zfile = ZipFile("sinonyms_compressed.zip")
    zfile.extractall()
    zfile.close()
    os.remove("sinonyms_compressed.zip")

with open('sinonyms_compressed.pickle', 'rb') as inp:
    sinonyms, syn_id2word = pickle.load(inp)

variants = pd.read_csv(
    "data/variants_clear_sorted.csv",
    sep=';',
    index_col="Unnamed: 0"
)
variants["variants"] = variants["variants"].apply(
    literal_eval
)

word_dict = pd.read_csv(
    "data/brown_corpus_tags.csv",
    sep=';',
    index_col="Unnamed: 0"
)["0"].apply(literal_eval).to_dict()

pos_dict = {
    tag: set() for word, tags in word_dict.items() for tag in tags
}
for tag in pos_dict:
    for word, tags in word_dict.items():
        if tag in tags:
            pos_dict[
                tag
            ].add(
                word
            )
pos_dict = {key: list(val) for key,val in pos_dict.items()}

word_dict = defaultdict(list, word_dict)

with open("data/freqdict.json", 'r', encoding='utf-8') as inp:
    freqdict = json.load(inp)
freqdict = defaultdict(lambda: 1, freqdict)

def get_word2vec_model() -> KeyedVectors:
    if not os.path.exists("gensim_models/skipgram_wikipedia_no_lemma"):
        download_word2vec_model()
    word2vec = KeyedVectors.load_word2vec_format(
        "gensim_models/skipgram_wikipedia_no_lemma/model.bin",
        binary=True
    )
    return word2vec


def get_bert_embedder() -> BertEmbedder:
    bert = BertEmbedder("bert-base-cased")
    return bert
