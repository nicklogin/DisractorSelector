import re
import pandas as pd
import numpy as np
import torch as tt
import scipy.spatial
import csv
import json
import os
import nltk
import json

from typing import List, Tuple, Union, Dict, Any

from transformers import BertTokenizer, BertModel
from ast import literal_eval
from tqdm import tqdm_notebook

from gensim.models import KeyedVectors
from collections import defaultdict


class BertEmbedder:
    def __init__(self, model_name: str):
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def _process_data(self, data: Union[str, List[str]]) -> tt.Tensor:
        tokenized = self.tokenizer(data, return_tensors="pt")
        with tt.no_grad():
            output = self.model(**tokenized)
        h = output.last_hidden_state
        return h, tokenized

    def embed_sentence(self, sent: str) -> List[float]:
        h, _ = self._process_data(sent)
        h_mean = h[0].mean(axis=0).numpy().tolist()
        return h_mean

    def embed_mask_token(self, sent: str) -> List[float]:
        h, tokenized = self._process_data(sent)
        h, tokenized = h[0], tokenized["input_ids"][0]
        mask_index = tokenized.numpy().tolist().index(
            self.tokenizer.mask_token_id
        )
        mask_embedding = h[mask_index].numpy().tolist()
        return mask_embedding

    def embed_tokens(self, sent: str) -> List[Tuple[str, float]]:
        h, tokenized = self._process_data(sent)
        h, tokenized = h[0], tokenized["input_ids"][0]
        tokens = self.tokenizer.convert_ids_to_tokens(tokenized)

        output = []

        for token, vector in zip(tokens, h):
            output.append((token, vector.numpy().tolist()))

        return output


with open("freqdict.json", 'r', encoding='utf-8') as inp:
    freqdict = json.load(inp)
freqdict = defaultdict(lambda: 1, freqdict)
word2vec = KeyedVectors.load_word2vec_format(
    "gensim_models/skipgram_wikipedia_no_lemma/model.txt"
)
bert = BertEmbedder("bert-base-cased")
variants = pd.read_csv(
    "variants_clear_sorted.csv", sep=';', index_col="Unnamed: 0"
)
variants["variants"] = variants["variants"].apply(
    literal_eval
)


def process_entry(
    masked_sent: str,
    right_answer: str,
    distractors: List[str]
) -> List[Dict[str, Any]]:
    output_dicts = []

    bert_masked = bert.embed_mask_token(
        masked_sent
    )
    bert_sent = bert.embed_sentence(
        masked_sent.replace("[MASK]", right_answer)
    )
    wvc = word2vec[right_answer]

    try:
        freq_corr = variants.loc[right_answer]["Count"]
    except KeyError:
        freq_corr = freqdict[right_answer]

    freq_corr_corp = 1

    for distractor in distractors:
        output_dict = {
            "freq_corr": freq_corr,
            "freq_corr_corp": freq_corr_corp
        }
        for i in range(len(wvc)):
            output_dict[f"wvc_{i}"] = wvc[i]
        for i in range(len(bert_masked)):
            output_dict[f"bm_{i}"] = bert_masked[i]
        for i in range(len(bert_sent)):
            output_dict[f"bs_{i}"] = bert_sent[i]
        # wve
        wve = word2vec[distractor]
        for i in range(len(wve)):
            output_dict[f"wve_{i}"] = wve[i]
        # freq_err_corr
        try:
            output_dict["freq_err_corr"] = variants.loc[
                right_answer
            ]["variants"][distractor]
        except KeyError:
            output_dict["freq_err_corr"] = 1
        # freq_err_corp
        output_dict["freq_err_corp"] = freqdict[distractor]
        output_dicts.append(output_dict)

    return output_dicts
