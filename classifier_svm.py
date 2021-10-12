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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from spacy.language import Language
from spacy.tokens.doc import Doc
from typing import List, Tuple, Set
import utils

# Concurrency
from joblib import Parallel, delayed, dump
from functools import partial
from multiprocessing import cpu_count

# ML
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight


class ConcurrentPreprocessor:
    """
    Concurrently preprocess text for downstream ML model training
    """

    def __init__(self, nlp: Language) -> None:
        self.nlp = nlp

    def run(
        self, texts: List[str], stopwords: Set[str], chunksize: int = 200
    ) -> List[str]:
        """
        Run concurrent jobs to preprocess a sequence of texts for downstream training
        """
        n_jobs = cpu_count() - 1  # Avoid max core usage
        executor = Parallel(
            n_jobs=n_jobs, backend="multiprocessing", prefer="processes"
        )
        do = delayed(partial(self.process_chunk, stopwords))
        tasks = (
            do(chunk) for chunk in self._chunker(texts, len(texts), chunksize=chunksize)
        )
        result = self._flatten(executor(tasks))
        return result

    def process_chunk(self, stopwords: Set[str], texts: List[str]) -> List[List[str]]:
        """
        Apply a spaCy language modelling pipeline and process a chunk of text
        """
        preproc_pipe = []
        for doc in self.nlp.pipe(texts, batch_size=20):
            preproc_pipe.append(self.lemmatize(doc, stopwords))
        return preproc_pipe

    def lemmatize(self, doc: Doc, stopwords: Set[str]) -> List[str]:
        """
        Perform lemmatization and stopword removal on clean text input
        """
        lemma_list = [
            tok.lemma_
            for tok in doc
            if tok.is_alpha and tok.text.lower() not in stopwords
        ]
        return lemma_list

    @staticmethod
    def _chunker(iterable: List[str], total_length: int, chunksize: int):
        """Divide an iterable into chunks that can be worked on concurrently"""
        return (
            iterable[pos : pos + chunksize] for pos in range(0, total_length, chunksize)
        )

    @staticmethod
    def _flatten(list_of_lists: List[List[str]]) -> List[str]:
        """Flatten a list of lists to a single, combined list"""
        return [item for sublist in list_of_lists for item in sublist]


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


def get_stopwords(filename: str = "stopwords.txt") -> Set[str]:
    """
    Read in stopwords from a newline-separated text file
    """
    with open(filename) as f:
        stopwords = set(sorted([word.strip() for word in f]))
    return stopwords


def transform_data(
    nlp: Language, data_file: str = "data.json", stopword_file: str = "stopwords.txt"
) -> pd.DataFrame:
    """
    Read in the data JSON and transform it to contain a column with lemmatized text
    """
    df = utils.read_data(data_file)
    stopwords = get_stopwords(stopword_file)
    processor = ConcurrentPreprocessor(nlp)

    # Concatenate title and abstract text prior to training
    df["text"] = df.apply(lambda x: f'{x["title"]}. {x["abstract"]}', axis=1)
    df["list_lemmas"] = processor.run(df["text"], stopwords)
    # Convert list of lemmas to a string, joined by spaces
    df["lemmas"] = df["list_lemmas"].str.join(" ")
    return df


if __name__ == "__main__":
    # Load spaCy language model for lemmatization
    nlp = spacy.load("en_core_web_sm")

    print("Transforming and lemmatizing data")
    data_df = transform_data(nlp, data_file="data_ipgb20201229_wk52.jsonl")
    svm = LinearSVM(data_df)
    model = svm.train_and_evaluate()
    # Dump model to disk using joblib
    dump(model, "svm_model.joblib")
