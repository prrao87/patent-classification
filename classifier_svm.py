"""
Code to train a linear Support Vector Machine (SVM) model in scikit-learn. The goal is
to be able to classify a patent based on the text in its title/abstract, into one of 8
top-level section labels. 
The section labels are as follows:
A, B, C, D, E, F, G, H

A: Human necessities
B: Performing operations; transporting
C: Chemistry; metallurgy
D: Textiles; paper
E: Fixed constructions
F: Mechanical engineering; lighting; heating; weapons; blasting
G: Physics
H: Electricity
"""
import argparse
from pathlib import Path
from typing import List, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

import utils

tqdm.pandas(desc="Lemmatization progress")

# ML
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight


def get_stopwords(filename: str = "stopwords.txt") -> Set[str]:
    """
    Read in stopwords from a newline-separated text file
    """
    with open(filename) as f:
        stopwords = set(sorted([word.strip() for word in f]))
    return stopwords


def lemmatize(text: str) -> List[str]:
    """Perform lemmatization and stopword removal in the clean text
    Returns a list of lemmas
    """
    doc = nlp(text)
    lemma_list = [
        str(tok.lemma_).lower()
        for tok in doc
        if tok.is_alpha and tok.text.lower() not in STOPWORDS
    ]
    return lemma_list


class LinearSVM:
    """
    Train a linear Support Vector Machine (SVM) using a scikit-learn pipeline
    """

    def __init__(self, data_df: pd.DataFrame) -> None:
        self.data = data_df
        assert "lemmas" in self.data.columns
        self.pipeline = Pipeline(
            [
                ("vect", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                (
                    "clf",
                    SGDClassifier(
                        loss="modified_huber",
                        penalty="l2",
                        alpha=5e-4,
                        random_state=42,
                        max_iter=100,
                        learning_rate="optimal",
                        tol=None,
                    ),
                ),
            ]
        )

    def train_and_evaluate(self) -> Pipeline:
        X_train, X_test, y_train, y_test = self.split_data()
        class_weights = self.compute_class_weights(y_train)
        print("Beginning training...")
        # Map class weights onto train DataFrame as a new column
        weights_df = self.data.loc[X_train.index]["label"].map(class_weights)
        model = self.pipeline.fit(X_train, y_train, clf__sample_weight=weights_df)
        # model = self.pipeline.fit(X_train, y_train)

        print("Finished training model.")
        preds = self.predict(model, X_test)
        # Print model performance
        utils.model_performance(y_test, preds)
        # Plot model performance as a confusion matrix
        self._plot_performance(y_train, y_test, preds)
        return model

    def split_data(self) -> Tuple[pd.DataFrame, ...]:
        train_test_data = train_test_split(
            self.data["lemmas"], self.data["label"], test_size=0.2, random_state=344535
        )
        return train_test_data

    def compute_class_weights(self, y_train) -> pd.DataFrame:
        """Compute class weights from the data for cost-sensitive weighting"""
        classes = np.unique(y_train)
        class_weights = class_weight.compute_class_weight(
            "balanced", classes=classes, y=y_train
        )
        final_class_weights = dict(zip(classes, class_weights))
        return final_class_weights

    def predict(self, model: Pipeline, X_test: pd.DataFrame) -> pd.Series:
        preds = model.predict(X_test)
        return preds

    @staticmethod
    def _plot_performance(y_train: pd.Series, true: pd.Series, pred: pd.Series) -> None:
        # Plot confusion matrix
        fig, _ = utils.plot_confusion_matrix(
            true, pred, classes=np.unique(y_train), normalize=True
        )
        fig.savefig("confusion_matrix.png")


def transform_data(data_file: str) -> pd.DataFrame:
    """
    Read in the data JSON and transform it to contain a column with lemmatized text
    """
    df = utils.read_data(data_file)
    # Concatenate title and abstract columns prior to training model
    df["text"] = df.apply(lambda x: f'{x["title"]}. {x["abstract"]}', axis=1)
    # Lemmatize text for better feature extraction
    df["list_lemmas"] = df["text"].progress_apply(lemmatize)
    # # Convert list of lemmas to a string, joined by spaces
    df["lemmas"] = df["list_lemmas"].str.join(" ")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SVM classifier trainer")
    parser.add_argument("--file", "-f", default="data.jsonl", help="Path to clean JSON line-delimited (jsonl) file")
    args = vars(parser.parse_args())
    if not Path(args["file"]).is_file():
        raise parser.error(f"\n{args['file']} not found -- please check that it exists and specify as input to script with the -f argument.")

    # Load spaCy language model for lemmatization
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
    nlp.add_pipe("sentencizer")
    STOPWORDS = get_stopwords("stopwords.txt")

    print("Transforming and lemmatizing data")
    data_df = transform_data(data_file=args['file'])
    svm = LinearSVM(data_df)
    model = svm.train_and_evaluate()
    # Dump model to disk using joblib
    joblib.dump(model, "svm_model.joblib")
