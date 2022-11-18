"""
Train model demande attestation
Refer to use case Reponse automatiques.

Be carefull to set the following environments variables (in bashrc for instance):
- CUDA_VISIBLE_DEVICES
- TOKENIZERS_PARALLELISM
"""
import os.path as op
from typing import Any, Callable, Dict, List

import hydra
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import transformers
from seqeval.metrics import classification_report
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mlflow_utils import log_hydra_params
from src.model import NERModel
from src.model.architectures import configure_optimizers
from src.model.datasets import NERDataset

tqdm.pandas()


def train_model(
    model: NERModel,
    train_data_loader: DataLoader,
    valid_data_loader: DataLoader,
    epochs: int,
    num_labels: int,
    loss_fn: Callable,
    tensorboard_writer: SummaryWriter,
) -> None:
    """
    Method to launch the model training.

    Parameters
    ----------
    model: NERModel
    train_data_loader: DataLoader
    valid_data_loader: DataLoader
    epochs: int
        Number of epochs in training
    num_labels: int
        Number of ner tags
    loss_fn: Callable
        Loss function used
    tensorboard_writer: SummaryWriter
        tensorboard_writer to log metrics
    """
    # Initialize training mode
    model.initialize_training(loss_fn=loss_fn)

    # Train loop
    total_iter = len(train_data_loader)
    for epoch in range(epochs):
        model.train()
        train_loss: float = 0
        with tqdm(train_data_loader, total=total_iter) as tqdm_iterator:
            for batch in tqdm_iterator:
                loss = model.training_step(batch=batch, num_labels=num_labels)

                # Logs
                train_loss += loss.item()
                logs_info = {"epoch": epoch, "loss": loss.item()}
                tqdm_iterator.set_postfix(logs_info)

                metrics_dict = validate_model(model, valid_data_loader, num_labels)

                # Logs
                train_loss = train_loss / total_iter
                logs_info = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **metrics_dict,
                }
                tqdm_iterator.set_postfix(logs_info)

                # Make sure logs apparears in ui
                if tensorboard_writer is not None:
                    for log, value in logs_info.items():
                        tensorboard_writer.add_scalar(log, value)
                    tensorboard_writer.flush()


def validate_model(model: NERModel, data_loader: DataLoader, num_labels: int) -> Dict[str, float]:
    """
    Evaluate on test data

    Parameters
    ----------
    model: NERModel
    data_loader: DataLoader
    num_labels: int
        Number of ner tags
    """
    model.eval()
    final_outputs = []
    with torch.no_grad():
        for batch in data_loader:

            outputs = model.validation_step(batch=batch, num_labels=num_labels)

            final_outputs.append(outputs)
        metrics_dict = model.validation_epoch_end(final_outputs)
    return metrics_dict


def get_offset_mapping(sentence: List[str], tokenizer: Any, max_len: int) -> List[int]:
    """Create an offset mapping which indicate the wordpieces of the byte paire encoding"""
    # Initialize offset_mapping to empty list
    offset_mapping = []

    for token in sentence:
        inputs = tokenizer(
            token,
            truncation=True,
            max_length=max_len,
        )
        # remove special tokens <s> and </s>
        ids_ = inputs["input_ids"][1:-1]

        # Consider only the first piece of the word
        offset_mapping.append(1)

        # Other pieces will be ignore to reconstruct the right prediction
        input_len = len(ids_)
        offset_mapping.extend([0] * (input_len - 1))

    # Reconstruct specials tokens at start and end
    offset_mapping = offset_mapping[: max_len - 2]
    offset_mapping = [0] + offset_mapping + [0]

    # Pad to the max len
    padding_len = max_len - len(offset_mapping)
    offset_mapping = offset_mapping + ([0] * padding_len)

    return offset_mapping


@hydra.main(config_path="../conf", config_name="train")
def main(cfg: Any) -> None:
    """
    Main function that train the model
    """
    extra_params = {"torch_version": torch.__version__, "transformers_version": transformers.__version__}

    # Setup MLflow
    mlflow.set_tracking_uri()
    mlflow.set_experiment(experiment_name=cfg.experiment_name)
    mlflow.pytorch.autolog(log_every_n_step=1, log_models=True)

    # Set Tensorboard saving directory
    # By default hydra will log everything in output/date/time
    # This is not handy when we want to compare several runs
    tensorboard_path = f"../../../tensorboard_runs/{cfg.experiment_name}/"

    # Load train data
    data = hydra.utils.instantiate(cfg.data, _convert_="all")

    if cfg.test_mode:
        cfg.fit.epochs = 1

    # Instantiate tokenizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer, _convert_="all")

    # Construct Dataset
    train_dataset = NERDataset(
        texts=data["train"]["tokens"],
        labels=data["train"]["ner_tags"],
        tokenizer=tokenizer,
        max_len=cfg.seq_max,
        loss_ignore_index=cfg.loss_ignore_index,
        propagate_label_to_word_pieces=cfg.propagate_label_to_word_pieces,
    )
    valid_dataset = NERDataset(
        texts=data["valid"]["tokens"],
        labels=data["valid"]["ner_tags"],
        tokenizer=tokenizer,
        max_len=cfg.seq_max,
        loss_ignore_index=cfg.loss_ignore_index,
        propagate_label_to_word_pieces=cfg.propagate_label_to_word_pieces,
    )
    test_dataset = NERDataset(
        texts=data["test"]["tokens"],
        labels=data["test"]["ner_tags"],
        tokenizer=tokenizer,
        max_len=cfg.seq_max,
        loss_ignore_index=cfg.loss_ignore_index,
        propagate_label_to_word_pieces=cfg.propagate_label_to_word_pieces,
    )

    # Construct DataLoader
    train_data_loader = train_dataset.get_data_loader(
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_data,
        num_workers=cfg.num_data_workers,
    )
    valid_data_loader = valid_dataset.get_data_loader(
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_data,
        num_workers=cfg.num_data_workers,
    )
    test_data_loader = test_dataset.get_data_loader(
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_data,
        num_workers=cfg.num_data_workers,
    )

    # Instantiate model
    model = hydra.utils.instantiate(cfg.model, _convert_="all")

    # Configure optimizer
    model.optimizer, model.scheduler = configure_optimizers(
        model=model,
        weight_decay=cfg.weight_decay,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        num_train_steps=int(len(data["train"]["tokens"]) / cfg.train_batch_size * cfg.epochs),
    )

    # Train & Evaluate model
    with mlflow.start_run(run_name=cfg.model_name), SummaryWriter() as tensorboard_writer:
        log_hydra_params(cfg)
        mlflow.log_params(extra_params)

        train_model(
            model,
            train_data_loader=train_data_loader,
            valid_data_loader=valid_data_loader,
            epochs=cfg.fit.epochs,
            num_labels=pd.Series(train_dataset["train"]["ner_tags"]).explode().unique(),
            loss_fn=hydra.utils.instantiate(cfg.fit.loss_fn, _convert_="all"),
            tensorboard_writer=tensorboard_writer,
        )

        # Evaluate model predictions
        evaluate_reports = []
        for threshold in np.linspace(0, 1, 11):
            # Run word pieces predictions
            wp_preds, wp_probas = model.predict(test_data_loader, progress_bar=True)

            # Get the token prediction based on word piece
            offset_mapping = [
                get_offset_mapping(sentence=sentence, tokenizer=tokenizer, max_len=cfg.seq_max)
                for sentence in test_dataset["test"]["tokens"]
            ]

            # Get predictions
            y_preds = []
            y_probas = []
            for sentence_pred, sentence_proba, sentence_mapping in zip(wp_preds, wp_probas, offset_mapping):
                y_pred = []
                y_proba = []
                for token_pred, token_proba, mapping in zip(sentence_pred, sentence_proba, sentence_mapping):
                    if mapping == 1:
                        y_pred.append(token_pred)
                        y_proba.append(token_proba)
                y_preds.append(y_pred)
                y_probas.append(y_proba)

            # Get true labels
            y_trues = test_dataset["test"]["ner_tags"]

            # Filter by threshold
            y_preds = [
                [y for y, y_prob in zip(y_pred, y_proba) if y_prob > threshold]
                for y_pred, y_proba in zip(y_preds, y_probas)
            ]
            y_trues = [
                [y for y, y_prob in zip(y_true, y_proba) if y_prob > threshold]
                for y_true, y_proba in zip(y_trues, y_probas)
            ]

            # Save classification report
            if all(len(y_true) > 0 for y_true in y_trues):
                report = classification_report(y_trues, y_preds)
                evaluate_reports.append(
                    (report, op.join(cfg.evaluate_report_path, f"report_threshold_{threshold}.txt"))
                )
            # If y_trues has len 0 we can stop iterating over threshold
            # Because higher threshold will lead to same result -> len(y_trues) = 0
            else:
                break

        # Log torch model
        mlflow.pytorch.log_model(artifact_path="model", pytorch_model=model)

        # Log tensorboard events
        mlflow.log_artifacts(tensorboard_path, artifact_path="tensorboard_logs")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
