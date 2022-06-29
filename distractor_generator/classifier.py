import pickle
import pandas as pd
import json

from typing import Any, List

from distractor_generator.utils import get_exec_time


@get_exec_time
def get_clf_and_cols(
    clf_path: str,
    cols_path: str
):
    with open(clf_path, 'rb') as inp:
        clf = pickle.load(inp)

    with open(cols_path, 'r', encoding='utf-8') as inp:
        cols = json.load(inp)

    return clf, cols


@get_exec_time
def classify_examples(
    examples: pd.DataFrame,
    clf: Any,
    cols: List
) -> List:
    examples = examples[cols]
    try:
        output = clf.predict(examples)
    except AttributeError:
        output = clf.predict_proba(examples)
    return output
