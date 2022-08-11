import pandas as pd
import json
import os

from typing import List, Dict, Any
from ast import literal_eval
from gensim.models import KeyedVectors
from collections import defaultdict
from tqdm import tqdm

from distractor_generator.utils import get_exec_time, download_word2vec_model
from distractor_generator.bert_embedder import BertEmbedder


with open("freqdict.json", 'r', encoding='utf-8') as inp:
    freqdict = json.load(inp)
freqdict = defaultdict(lambda: 1, freqdict)

if not os.path.exists("gensim_models/skipgram_wikipedia_no_lemma"):
    download_word2vec_model()
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


@get_exec_time
def batch_process_entries(
    masked_sents: List[str],
    right_answers: List[str],
    distractor_lists: List[List[str]],
    sent_ids: List[int] = None
):
    # optimize
    # оптимизация не требуется, большую часть времени занимает обработка предложений bert'ом
    if sent_ids is None:
        sent_ids = list(range(len(masked_sents)))
    output_dicts = []

    bert_masked_sents = bert.batch_embed_mask_tokens(masked_sents)
    bert_sents = bert.batch_embed_sentences(
        [
            sent.replace(
                "[MASK]", right_answer
            ) for sent, right_answer in zip(
                masked_sents, right_answers
            )
        ]
    )

    # print("Postprocessing examples ... ")
    for sent_id, bert_masked, bert_sent, right_answer, distractors in tqdm(zip(
        sent_ids,
        bert_masked_sents,
        bert_sents,
        right_answers,
        distractor_lists
    ), total=len(sent_ids)):
        wvc = word2vec[right_answer]

        try:
            freq_corr = variants.loc[right_answer]["Count"]
        except KeyError:
            freq_corr = freqdict[right_answer]

        freq_corr_corp = 1

        for distractor in distractors:
            output_dict = {
                "sent_id": sent_id,
                "variant": distractor,
                "freq_corr": freq_corr,
                "freq_corr_corp": freq_corr_corp,
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
