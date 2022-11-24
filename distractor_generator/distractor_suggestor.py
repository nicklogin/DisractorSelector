import pandas as pd
import numpy as np

import re
import os
import shutil
import json
import pickle

from typing import List, Tuple
from collections import defaultdict
from ast import literal_eval
from nltk import word_tokenize
from string import punctuation
from gensim.models import KeyedVectors

from zipfile import ZipFile

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


def is_aux_verb(word: str) -> bool:
    word = word.lower()

    aux_verbs = [
        "be", "being", "is", "are",
        "can", "could", "shall", "should",
        "do", "did", "done",
        "shall", "should", 
        "need", "ought",
        "might", "must", "may",
        "will", "would"
    ]
    
    if word in aux_verbs:
        return True
    return False


def is_negation(word_a: str, word_b: str) -> bool:
    # optimize
    # здесь регулярка всё замедляет, переписать без регулярки
    word_a = word_a.lower()
    word_b = word_b.lower()

    # if re.match(f"(un|im|in|non){re.escape(word_a)}", word_b):
    #     return True

    # if re.match(f"(un|im|in|non){re.escape(word_b)}", word_a):
    #     return True
    prefixes = ("un","im","in","non")

    if word_a in (prefix+word_b for prefix in prefixes):
        return True
    elif word_b in (prefix+word_a for prefix in prefixes):
        return True

    return False


def is_ungradable(word: str) -> bool:
    word = word.lower()
    vowels = "aeyuio"
    count_vowels = 0

    for char in word:
        if char in vowels:
            count_vowels += 1

    if count_vowels == 1 or (
        count_vowels == 2 and
        word.endswith("e")
    ):
        return True
    return False


def mask_after_degree(masked_sent: str) -> bool:
    masked_sent = masked_sent.lower()
    chunks = [
        "more [mask]",
        "less [mask]",
        "least [mask]",
        "most [mask]"
    ]
    for chunk in chunks:
        if chunk in masked_sent:
            return True
    return False


def is_indicator_name(word: str) -> bool:
    word = word.lower()

    indicator_names = [
        "amount",
        "quantity",
        "number",
        "level",
        "percentage",
        "degree",
        "rate",
        "share",
        "index",
        "figure",
        "proportion",
        "portion",
        "amounts",
        "quantities",
        "numbers",
        "levels",
        "percentages",
        "degrees",
        "rates",
        "shares",
        "indexes",
        "figures",
        "proportions",
        "portions"
    ]

    if word in indicator_names:
        return True

    return False


def split_masked_sent(
    masked_sent: str
) -> Tuple[str, str]:
    masked_sent = masked_sent.lower()
    left_context = [
        word for word in word_tokenize(
            masked_sent[:masked_sent.find('[mask]')]
        ) if word not in punctuation
    ]
    right_context = [word for word in word_tokenize(
        masked_sent[masked_sent.find('[mask]')+6:]
    ) if word not in punctuation
    ]
    return left_context, right_context


def is_context_word_copied(
    left_context: str,
    right_context: str,
    word: str,
    depth=1
) -> bool:
    left_context = left_context[-depth:]
    right_context = right_context[depth:]

    if word in right_context or word in left_context:
        return True
    return False


# def get_variant_count(word: str, x: str):
#     print(x, word)
#     vcount = variants.loc[word]["variants"].get(x)
#     print(str(vcount))
#     return vcount


# Function that suggests distractors:
def suggest_distractors_from_corpus(
    masked_sent: str,
    word: str
) -> List[str]:
    distractors = []

    # дистракторы из ошибок realec:
    if word in variants.index:
        distractors += [
            key for key, val in variants.loc[
                word
            ]["variants"].items() if len(
                set(
                    word_dict[word]
                ) & set(
                    word_dict[key]
                )
            ) > 0
        ]
    
    left_context, right_context = split_masked_sent(masked_sent)

    distractors = list(
        set(
            d for d in distractors if d != word and
            {d, word} != {"life", "level"} and
            not is_aux_verb(d) and not is_negation(d, word) and
            not (is_ungradable(d) and mask_after_degree(masked_sent)) and
            not (is_indicator_name(word) and d in ("line", "lines")) and
            not is_context_word_copied(left_context, right_context, d)
        )
    )
    distractors = sorted(
        distractors,
        key=lambda x: variants.loc[word]["variants"].get(x),
        reverse=True
    )

    return distractors


def batch_add_distractors_from_corpus(
    sents: List[str],
    corrections: List[str]
) -> List[List[str]]:
    # optimize
    variants = []

    for sent, correction in zip(sents, corrections):
        distractors = suggest_distractors_from_corpus(sent, correction)
        variants.append(distractors)

    return variants


def batch_apply_vocab_filters(
    words: List[str],
    candidates: List[List[str]]
) -> List[List[str]]:
    candidates = [
        [
            candidate for candidate in candidate_list if
            not is_aux_verb(candidate) and
            not (is_indicator_name(word) and candidate in ("line", "lines")) and
            not is_negation(candidate, word)
        ] for word, candidate_list in zip(words, candidates)
    ]
    return candidates


def batch_apply_context_filters(
    candidates: List[List[str]],
    masked_sents: List[str]
):
    # 1 условие:
    candidates = [
        [
            candidate for candidate in candidate_list if
            not (is_ungradable(candidate) and mask_after_degree(masked_sent))
        ] for candidate_list, masked_sent in zip(candidates, masked_sents)
    ]

    #2 условие
    contexts = [split_masked_sent(masked_sent) for masked_sent in masked_sents]
    candidates = [
        [
            candidate for candidate in candidate_list if
            not is_context_word_copied(left_context, right_context, word)
        ] for candidate_list, (left_context, right_context) in zip(candidates, contexts)
    ]

    return candidates


def batch_add_distractors_from_word2vec(
    words: List[str],
    masked_sents: List[str],
    distractors: List[List[str]],
    min_candidates: int = 20
) -> List[List[str]]:
    candidates = [
        [syn_id2word[idx] for idx in sinonyms[word]] for word in words
    ]
    ## применить словарные фильтры
    candidates = batch_apply_vocab_filters(words, candidates)
    ## применить контекстуальные фильтры
    candidates = batch_apply_context_filters(candidates, masked_sents)
    # удалить существующие дистракторы (чтобы не записывать их два раза):
    candidates = [
        [candidate for candidate in candidate_list if candidate not in variant_list]
        for variant_list, candidate_list in zip(distractors, candidates)
    ]
    ## приплюснуть к существующим дистракторам
    candidates = [
        (variant_list + candidate_list)[:min_candidates]
        for variant_list, candidate_list in zip(distractors, candidates)
    ]
    ## насколько это ускорит работу модели - 
    ## не знаю, но надо попробовать
    return candidates


def batch_suggest_distractors(
    sents: List[str],
    corrections: List[str],
    n: int
) -> List[List[str]]:
    variants = batch_add_distractors_from_corpus(sents, corrections)

    variants = batch_add_distractors_from_word2vec(
        corrections,
        sents,
        variants,
        n
    )

    return variants
