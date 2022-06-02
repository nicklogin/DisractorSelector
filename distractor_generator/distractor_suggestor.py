import pandas as pd
import numpy as np

import re

from typing import List
from collections import defaultdict
from ast import literal_eval
from nltk import word_tokenize
from string import punctuation
from gensim.models import KeyedVectors

# Load data resources:
word2vec = KeyedVectors.load_word2vec_format(
    "gensim_models/skipgram_wikipedia_no_lemma/model.txt"
)

variants = pd.read_csv(
    "variants_clear_sorted.csv",
    sep=';',
    index_col="Unnamed: 0"
)
variants["variants"] = variants["variants"].apply(
    literal_eval
)

word_dict = pd.read_csv(
    "brown_corpus_tags.csv",
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


def is_aux_verb(word: str):
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


def is_negation(word_a: str, word_b: str):
    word_a = word_a.lower()
    word_b = word_b.lower()

    if re.match(f"(un|im|in|non){word_a}", word_b):
        return True

    if re.match(f"(un|im|in|non){word_b}", word_a):
        return True

    return False


def is_ungradable(word: str):
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


def mask_after_degree(masked_sent: str):
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


def is_indicator_name(word: str):
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


def is_context_word_copied(
    masked_sent: str,
    word: str,
    depth=1
):
    masked_sent = masked_sent.lower()
    left_context = [
        word for word in word_tokenize(
            masked_sent[:masked_sent.find('[mask]')]
        ) if word not in punctuation
    ][:-depth]
    right_context = [word for word in word_tokenize(
        masked_sent[masked_sent.find('[mask]')+6:]
    ) if word not in punctuation
    ][depth:]

    if word in right_context or word in left_context:
        return True
    return False


# Function that suggests distractors:
def suggest_distractors(
    masked_sent: str,
    word: str,
    min_candidates=20
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

    distractors = list(
        set(
            d for d in distractors if d != word and
            {d, word} != {"life", "level"} and
            not is_aux_verb(d) and not is_negation(d, word) and
            not (is_ungradable(d) and mask_after_degree(masked_sent)) and
            not (is_indicator_name(word) and d in ("line", "lines")) and
            not is_context_word_copied(masked_sent, d)
        )
    )
    distractors = sorted(
        distractors,
        key=variants.loc[word]["variants"].get,
        reverse=True
    )

    if len(distractors) > min_candidates:
        distractors = distractors[:min_candidates]
    elif len(distractors) < min_candidates:
        # добираем слова до минимума из word2vec:
        n_needed = min_candidates - len(distractors)

        # слова тех же возможных частей речи:
        pos_tags = word_dict[word]

        if pos_tags:
            candidates = [w for tag in pos_tags for w in pos_dict[tag]]
            candidates = list(
                set(
                    d for d in candidates if d in word2vec.key_to_index and
                    d not in distractors and
                    d != word and
                    {d, word} != {"life", "level"} and
                    d not in punctuation
                )
            )
            candidates = [
                d for d in candidates if
                not is_aux_verb(d) and
                not (is_indicator_name(word) and d in ("line", "lines"))
            ]

            candidates = [
                d for d in candidates if
                not (is_ungradable(d) and mask_after_degree(masked_sent)) and
                not is_context_word_copied(masked_sent, d)
            ]

            candidates = [
                d for d in candidates if
                not is_negation(d, word)
            ]

            order = word2vec.distances(word, candidates).argsort()
            candidates = np.array(candidates)[order]
            candidates = list(candidates[:n_needed])
            distractors += candidates

    return distractors


# print(suggest_distractors("possibilities"))
# print(suggest_distractors("number"))
# print(suggest_distractors("women"))
# print(suggest_distractors("conclusion"))
