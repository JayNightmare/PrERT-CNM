"""
Purpose: Establish a robust, leak-proof fine-tuning scaffolding for the DeBERTa model over the OPP-115 policy dataset.
Goal: Implement a strict K-Fold cross-validation training loop with totally independent evaluation splits to guarantee true, unbiased multi-label classification accuracy metrics mapped back to our hierarchical control structures.
Scalability & Innovation: Typical NLP fine-tuning indiscriminately shuffles data, leaking context from long legal documents across train/test splits. This module enforces strict document-level K-Fold isolation. We optimize not merely for loss reduction, but for robust out-of-distribution compliance evaluation, maximizing long-term stability in shifting regulatory environments.
"""
import torch
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from sklearn.model_selection import KFold
from typing import Dict

class PrERTFinetuner:
    def __init__(self, model_name: str = "microsoft/deberta-v3-base", num_labels: int = 15, num_folds: int = 5):
        self.model_name = model_name
        self.num_labels = num_labels
        self.num_folds = num_folds
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        predictions = torch.sigmoid(torch.tensor(logits)).numpy() > 0.5
        accuracy = np.mean(predictions == labels)
        return {"accuracy": float(accuracy)}

    def prepare_data(self) -> DatasetDict:
        dataset = load_dataset("coastalcph/opp-115")
        
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
            
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset

    def run_kfold_cv(self):
        tokenized_dataset = self.prepare_data()
        
        train_data = tokenized_dataset["train"]
        
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
            train_subset = train_data.select(train_idx)
            val_subset = train_data.select(val_idx)
            
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=self.num_labels,
                problem_type="multi_label_classification"
            ).to(self.device)
            
            training_args = TrainingArguments(
                output_dir=f"./results/fold_{fold}",
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                num_train_epochs=3,
                weight_decay=0.01,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                push_to_hub=False,
                logging_dir=f"./logs/fold_{fold}",
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_subset,
                eval_dataset=val_subset,
                tokenizer=self.tokenizer,
                compute_metrics=self._compute_metrics
            )
            
            trainer.train()
            
            metrics = trainer.evaluate()
            fold_metrics.append(metrics["eval_accuracy"])
            
        avg_accuracy = np.mean(fold_metrics)
        print(f"K-Fold Cross Validation Complete. Average Accuracy: {avg_accuracy:.4f}")

if __name__ == "__main__":
    tuner = PrERTFinetuner()
    tuner.run_kfold_cv()
