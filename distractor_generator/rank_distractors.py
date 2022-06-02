import numpy as np

from typing import List, Dict

from gensim.models import KeyedVectors

# Load data resources:
word2vec = KeyedVectors.load_word2vec_format(
    "gensim_models/skipgram_wikipedia_no_lemma/model.txt"
)


def rank_distractors(
    distractors: Dict[int, List[str]],
    corrections: List[str],
    sents: List[str]
) -> Dict[int, List[str]]:
    distractors_sorted = dict()

    for sent_id, distractor_list in distractors.items():
        correction = corrections[sent_id]

        distractor_list = np.array(
            distractor_list
        )[
            word2vec.distances(correction, distractor_list).argsort()
        ].tolist()
        distractors_sorted[sent_id] = distractor_list

    return distractors_sorted
