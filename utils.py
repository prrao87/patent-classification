import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


def read_data(filepath: str) -> pd.DataFrame:
    """
    Read in a JSON file with doc_id, title, abstract and section label.
    Assumes that there are newlines separating multiple JSON objects (jsonl)
    """
    df = pd.read_json(filepath, lines=True)
    return df


def plot_confusion_matrix(
    y_true,
    y_pred,
    classes=["A", "B", "C", "D", "E", "F", "G", "H"],
    normalize=False,
    cmap=plt.cm.YlOrBr,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (Adapted from scikit-learn docs).
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", origin="lower", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # Show all ticks
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # Label with respective list entries
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Set alignment of tick labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    return fig, ax


def model_performance(true: pd.Series, pred: pd.Series) -> None:
    """
    Print out relevant model performance metrics
    """
    accuracy = accuracy_score(true, pred, normalize=True)
    macro_f1 = f1_score(true, pred, average="macro")
    micro_f1 = f1_score(true, pred, average="micro")
    weighted_f1 = f1_score(true, pred, average="weighted")
    # TODO: Use logger instead of print statements
    print(
        f"Macro F1: {100*macro_f1:.3f} %\nMicro F1: {100*micro_f1:.3f} %\nWeighted F1: {100*weighted_f1:.3f} %"
    )
    print(f"Accuracy: {100*accuracy:.3f} %")