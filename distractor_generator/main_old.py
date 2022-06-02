import pandas as pd

from distractor_generator.distractor_suggestor import suggest_distractors
from distractor_generator.example_processor import process_entry
from distractor_generator.classifier import classify_examples
from distractor_generator.rank_distractors import rank_distractors

df = pd.read_csv("gold_standard.csv", sep=';', index_col="Unnamed: 0")
df = df.loc[(
    (
        df["Distractor 1"] != "Delete"
    ) & (
        df["Distractor 2"] != "Delete"
    ) & (
        df["Distractor 3"] != "Delete"
    )
)]
sents = df["Masked_sentence"].tolist()
corrections = df["Right_answer"].tolist()

output_df = []

for idx, (sent, correction) in enumerate(zip(sents, corrections)):
    variants = suggest_distractors(correction, min_candidates=12)
    examples = process_entry(sent, correction, variants)
    for example, variant in zip(examples, variants):
        example["sent_id"] = idx
        example["variant"] = variant

    output_df += examples

output_df = pd.DataFrame(output_df)
output_df.to_csv("examples.csv", sep=';')
targets = classify_examples(
    output_df.drop(["sent_id", "variant"], axis=1)
)
output_df["target"] = targets

distractors = output_df.loc[
    output_df["target"] >= 0.5
].groupby("sent_id")["variant"].unique().apply(lambda x: x.tolist()).to_dict()
distractors = {int(key): val for key, val in distractors.items()}

print("Ranking distractors ... ")
distractors = rank_distractors(distractors, corrections, sents)
print("Distractors ranked successfully")

none_to_list = lambda x: [] if x is None else x

output = [
    (
        sent,
        correction,
        none_to_list(distractors.get(sent_id))
    ) for sent_id, (sent, correction) in enumerate(zip(sents, corrections))
]

output_df = [
    {
        "sent": sent,
        "right_answer": correction,
        "predicted_distractor_1": distractors[0] if len(distractors) > 0 else None,
        "predicted_distractor_2": distractors[1] if len(distractors) > 1 else None,
        "predicted_distractor_3": distractors[2] if len(distractors) > 2 else None,
        "all_predicted_distractors": distractors,
        "distractor_1": d1,
        "distractor_2": d2,
        "distractor_3": d3
    } for (sent, correction, distractors), d1, d2, d3 in zip(
        output,
        df["Distractor 1"],
        df["Distractor 2"],
        df["Distractor 3"]
    )
]
output_df = pd.DataFrame(output_df)
output_df.to_csv("gold_standard_output.csv", sep=';')

c1 = 0
c2 = 0
c3 = 0

for idx, row in output_df.iterrows():
    if {
        row["predicted_distractor_1"],
        row["predicted_distractor_2"],
        row["predicted_distractor_3"]
    } == {
        row["distractor_1"],
        row["distractor_2"],
        row["distractor_3"]
    }:
        c1 += 1

    c2 += len(
        {
            row["predicted_distractor_1"],
            row["predicted_distractor_2"],
            row["predicted_distractor_3"]
        } & {
            row["distractor_1"],
            row["distractor_2"],
            row["distractor_3"]
        }
    )

    c3 += len(
        set(
            row["all_predicted_distractors"]
        ) & {
            row["distractor_1"],
            row["distractor_2"],
            row["distractor_3"]
        }
    )

# процент совпавших сетов:
print(c1/len(output_df))

# процент угаданных дистракторов:
print(c2/(len(output_df)*3))

# как в (Ha, Yaneva 2018)
print(c3/(len(output_df)*3))
