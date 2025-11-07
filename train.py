"""
Train a RoBERTa-based classifier for book-text matching.

The model learns to predict whether two text chunks come from the same book (label=1) or different books (label=0).
"""
import os
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np


def tokenize_function(examples, tokenizer):
    """Tokenize pairs of texts."""
    return tokenizer(
        examples['text1'],
        examples['text2'],
        truncation=True,
        padding='max_length',
        max_length=512
    )


def compute_metrics(eval_pred):
    """Compute accuracy, F1, precision, recall, and AUC."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    
    # AUC requires probability scores, not predictions
    predictions_proba = torch.softmax(torch.from_numpy(eval_pred.predictions), dim=1)[:, 1].numpy()
    auc = roc_auc_score(labels, predictions_proba)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }


def train(
    model_name: str = 'roberta-base',
    output_dir: str = 'models/book_matcher',
    data_dir: str = 'data/processed',
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_steps: int = 500
):
    """Train the model."""
    print(f"Loading datasets from {data_dir}...")
    from datasets import load_from_disk
    datasets = load_from_disk(data_dir)
    
    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_datasets = datasets.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['text1', 'text2']
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        eval_strategy='steps',
        eval_steps=500,
        save_strategy='steps',
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        report_to='none',
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        remove_unused_columns=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(tokenized_datasets['test'])
    print(f"Test results: {test_results}")
    
    # Save final model
    trainer.save_model(f'{output_dir}/final')
    tokenizer.save_pretrained(f'{output_dir}/final')
    print(f"\nModel saved to {output_dir}/final")
    
    return trainer, test_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train book-text matching classifier')
    parser.add_argument('--model', type=str, default='roberta-base', help='Base model name')
    parser.add_argument('--output', type=str, default='models/book_matcher', help='Output directory')
    parser.add_argument('--data', type=str, default='data/processed', help='Data directory')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    
    args = parser.parse_args()
    
    train(
        model_name=args.model,
        output_dir=args.output,
        data_dir=args.data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

