import os
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from datasets import load_dataset, DatasetDict, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from typing import Dict


def create_train_valid_test(data_json: str) -> DatasetDict:
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


def compute_metrics(pred) -> Dict[str, float]:
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def get_encoded_data(tokenizer: AutoTokenizer, textcol: str = "text") -> DatasetDict:
    def tokenize(batch):
        tokens = tokenizer(batch[textcol], truncation=True, padding=True)
        tokens["label"] = labels.str2int(batch["label"])
        return tokens

    encoded_dataset = dataset.map(tokenize, batched=True, load_from_cache_file=False)
    encoded_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "label"]
    )
    return encoded_dataset


def set_tokenizer(batch_size: int = 32) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint,
        use_fast=True,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return tokenizer


def train() -> None:
    args = TrainingArguments(
        "transformers-classification",
        evaluation_strategy="epoch",
        learning_rate=4e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        # fp16=True,   # If true, enable mixed precision (16 bit + 32 bits)
        # fp16_opt_level="O1",
        # fp16_backend="auto",  # Either amp or apex fp16 backends
        warmup_ratio=0.25,  # Slanted triangular learning rate ramp up/down (0.2-0.3 gave best results on multiple runs)
        weight_decay=0.001,  # L2 regularization
        # load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    tokenizer = set_tokenizer(batch_size=32)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=len(classes)
    )
    encoded_dataset = get_encoded_data(tokenizer, textcol="text")
    trainer = Trainer(
        model,
        args=args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["valid"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    # Train & evaluate
    trainer.train()
    trainer.evaluate()
    os.makedirs("pytorch_model", exist_ok=True)
    trainer.save_model(output_dir="pytorch_model")


if __name__ == "__main__":
    # Define class labels in the dataset
    classes = ["A", "B", "C", "D", "E", "F", "G", "H"]
    # For encoding
    labels = ClassLabel(names=classes)
    dataset = create_train_valid_test("../data_ipgb20201229_wk52_combined.jsonl")
    # Use DistilBERT base uncased model
    model_checkpoint = "distilbert-base-uncased"
    train()
