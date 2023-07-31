from typing import List, Tuple
from nltk import word_tokenize
from string import punctuation

from distractor_generator.resources import sinonyms, syn_id2word
from distractor_generator.resources import word_dict
from distractor_generator.resources import variants


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
    word_a = word_a.lower()
    word_b = word_b.lower()

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


def suggest_distractors_from_corpus(
    masked_sent: str,
    word: str
) -> List[str]:
    distractors = []

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
    masked_sents: List[str],
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
            not is_context_word_copied(left_context, right_context, candidate)
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
        [
            syn_id2word[idx] for idx in sinonyms[word.lower()]
        ] if word.lower() in sinonyms
        else []
        for word in words
    ]

    candidates = [
        [cand.title() for cand in cand_list] if word.istitle()
        else [cand.upper() for cand in cand_list] if word.isupper()
        else cand_list
        for word, cand_list in zip(words, candidates)
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
