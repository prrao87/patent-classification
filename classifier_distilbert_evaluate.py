import numpy as np
import pandas as pd
import torch
from typing import List, Dict
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
)
from datasets import load_dataset, DatasetDict
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
        "Lookup label name from PyTorch integer prediction value"
        label_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}
        return label_map[pred]

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


def create_train_valid_test(data_json: str) -> DatasetDict:
    """
    IMPORTANT: Use the SAME train/valid/test split with the same random seed as used in
    the training script.
    """
    dataset = load_dataset(
        "json",
        data_files={"data": data_json},
        split={"train": "data[:100%]"},
    )
    # 70% train, 30% test + validation
    train_testvalid = dataset["train"].train_test_split(test_size=0.3, seed=344535)
    # Further split the (test + validation) data into 20% test + 10% valid
    test_valid = train_testvalid["test"].train_test_split(
        test_size=0.66667, seed=344535
    )
    # Gather everything to a single DatasetDict
    dataset_dict = DatasetDict(
        {
            "train": train_testvalid["train"],
            "test": test_valid["test"],
            "valid": test_valid["train"],
        }
    )
    return dataset_dict


def main(path_to_test_data: str, model: DistilbertOnnxPredictor) -> None:
    print("Encoding train/valid/test data as per training script's split...")
    dataset = create_train_valid_test(path_to_test_data)
    X_test = dataset["test"]["text"]
    y_train = dataset["train"]["label"]
    y_test = dataset["test"]["label"]
    # Make model predictions using the ONNX predictor
    print("Making predictions on test set...")
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
