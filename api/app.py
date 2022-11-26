import pandas as pd

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Dict, Optional

from distractor_generator.distractor_suggestor import batch_suggest_distractors
from distractor_generator.example_processor import batch_process_entries
from distractor_generator.classifier import get_clf_and_cols, classify_examples


clfs = [
    "CatBoostFeatDrop",
    "CatBoostVecsOnly",
    "RandomForestFreqsOnly",
    "XGBAllFeats",
    "None"
]


class Example(BaseModel):
    id: int
    masked_sent: str
    correction: str


class ProcessedExample(Example):
    variants: List[str]


app = FastAPI(
    title="Distractor Suggestor",
    version="0.1.0"
)


@app.post("/")
def get_distractors(
    examples: List[Example],
    n: Optional[int] = Query(4),
    clf: Optional[str] = Query("XGBAllFeats", enum=clfs)
) -> List[ProcessedExample]:
    sents = [example.masked_sent for example in examples]
    corrections = [example.correction for example in examples]

    variants = batch_suggest_distractors(sents, corrections, n)

    if clf == "None":
        output = [
            ProcessedExample(
                id=example.id,
                masked_sent=example.masked_sent,
                correction=example.correction,
                variants=variant_list
            ) for example, variant_list in zip(examples, variants)
        ]
        return output

    classifier, columns = get_clf_and_cols(
        f"{clf}/clf.pkl",
        f"{clf}/cols.json"
    )
    output_df = batch_process_entries(sents, corrections, variants)

    output_df = pd.DataFrame(output_df)
    targets = classify_examples(
        output_df.drop(["sent_id", "variant"], axis=1),
        classifier,
        columns
    )
    output_df["target"] = targets
    output_df.to_csv("output_df.csv")

    distractors = output_df.loc[
        output_df["target"] >= 0.5
    ].groupby("sent_id")["variant"].unique().apply(
        lambda x: x.tolist()
    ).to_dict()
    distractors = {int(key): val for key, val in distractors.items()}

    for key in range(len(sents)):
        if key not in distractors:
            distractors[key] = []
    distractors = [distractors[key] for key in sorted(distractors)]

    output = [
        ProcessedExample(
            id=example.id,
            masked_sent=example.masked_sent,
            correction=example.correction,
            variants=variant_list
        ) for example, variant_list in zip(examples, distractors)
    ]

    return output

