import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Dict
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import utils


class DistilbertOnnxPredictor:
    def __init__(self, path_to_model: str, islower: bool = True) -> None:
        from transformers import AutoTokenizer

        model_checkpoint = (
            "distilbert-base-uncased" if islower else "distilbert-base-cased"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = self.create_model_for_provider(path_to_model)

    def _get_class_label(self, pred: int) -> str:
        "Reverse lookup label name from PyTorch integer prediction value"
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7}
        for key, value in label_map.items():
            if value == pred:
                return key

    def create_model_for_provider(self, model_path: str) -> InferenceSession:
        "Create ONNX model based on provider (we use CPU by default)"
        # Few properties that might have an impact on performance
        options = SessionOptions()
        options.intra_op_num_threads = 3
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load the model as a graph and prepare the backend
        session = InferenceSession(
            model_path, options, providers=["CPUExecutionProvider"]
        )
        session.disable_fallback()
        return session

    def get_onnx_inputs(self, text: str) -> Dict[List[int], List[int]]:
        "Input IDs after tokenization are provided as a numpy array"
        model_inputs = self.tokenizer(
            text, max_length=512, padding=True, truncation=True, return_tensors="pt"
        )
        inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
        return inputs_onnx

    def predict_proba(self, text: str) -> np.ndarray:
        """
        Returns a numpy array of dimension (M x N) where M = no. of text samples
        and N = no. of class labels
        The value at each index represents the probability of each class label.
        """
        inputs_onnx = self.get_onnx_inputs(text)
        raw_outputs = self.model.run(None, inputs_onnx)
        probs = torch.softmax(torch.tensor(raw_outputs[0]), dim=1)
        probs = np.array(probs)[0]  # convert to numpy array
        return probs

    def predict(self, text: str) -> int:
        """
        0: anxiety, 1: depression, 2: positive mood, 3: negative mood
        """
        probs = self.predict_proba(text)
        pred = np.argmax(probs)
        class_label = self._get_class_label(pred)
        return class_label


def split_data(data_df: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    # Same random state as the SVM model!
    train_test_data = train_test_split(
        data_df["text"], data_df["label"], test_size=0.2, random_state=344535
    )
    return train_test_data


def main(path_to_test_data: str, model: DistilbertOnnxPredictor) -> None:
    df = utils.read_data(path_to_test_data)
    X_train, X_test, y_train, y_test = split_data(df)
    # Make model predictions using the ONNX predictor
    preds = pd.Series([model.predict(text) for text in tqdm(X_test)])
    # Plot confusion matrix
    fig, _ = utils.plot_confusion_matrix(
        y_test, preds, classes=np.unique(y_train), normalize=True
    )
    fig.savefig("confusion_matrix.png")
    utils.model_performance(y_test, preds)


if __name__ == "__main__":
    # Export the trained DistilBERT model in ONNX format to the following path
    path_to_model = "onnx_model/onnx_model-optimized-quantized"
    model = DistilbertOnnxPredictor(path_to_model, islower=True)

    # Path to test data
    path_to_test_data = "data_ipgb20201229_wk52_combined.jsonl"
    main(path_to_test_data, model)
