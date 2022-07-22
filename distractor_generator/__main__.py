import pandas as pd

from distractor_generator.distractor_suggestor import batch_suggest_distractors
from distractor_generator.example_processor import batch_process_entries
from distractor_generator.classifier import get_clf_and_cols, classify_examples

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--filename", type=str, default="gold_standard_input.csv"
    )
    parser.add_argument("--output_filename", type=str, default=None)
    parser.add_argument("--sep", type=str, default=";")
    parser.add_argument("--index_col", type=str, default="Unnamed: 0")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--no-clf", action="store_true", dest="no_clf")
    parser.add_argument(
        "--clf_path",
        type=str,
        default="XGBAllFeats/clf.pkl"
    )
    parser.add_argument(
        "--cols_path",
        type=str,
        default="XGBALLFeats/cols.json"
    )
    args = vars(parser.parse_args())

    classifier, columns = get_clf_and_cols(
        args["clf_path"],
        args["cols_path"]
    )

    if args["output_filename"] is None:
        output_filename = f"{args.get('filename').split('.')[0]}_output.csv"
    else:
        output_filename = args["output_filename"]

    df = pd.read_csv(
        args.get("filename"),
        sep=args.get("sep"),
        index_col=args.get("index_col")
    )
    # df = df.loc[(
    #     (
    #         df["Distractor 1"] != "Delete"
    #     ) & (
    #         df["Distractor 2"] != "Delete"
    #     ) & (
    #         df["Distractor 3"] != "Delete"
    #     )
    # )]
    sents = df["Masked_sentence"].tolist()
    corrections = df["Right_answer"].tolist()

    if args.get("no_clf"):
        variants = batch_suggest_distractors(sents, corrections, args.get('n'))
        df["variants"] = variants
        # df.to_csv(
        #     output_filename, sep=';'
        # )
    else:
        output_df = []
        variants = batch_suggest_distractors(sents, corrections, args.get('n'))
        output_df = batch_process_entries(sents, corrections, variants)

        output_df = pd.DataFrame(output_df)
        targets = classify_examples(
            output_df.drop(["sent_id", "variant"], axis=1),
            classifier,
            columns
        )
        output_df["target"] = targets

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
        df["variants"] = distractors
        df.to_csv(
            output_filename, sep=';'
        )
