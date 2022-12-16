"""
Train model ner

Be carefull to set the following environments variables (in bashrc for instance):
- CUDA_VISIBLE_DEVICES
- TOKENIZERS_PARALLELISM
"""
from typing import Any, Callable, Dict

import hydra
import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.mlflow_utils import log_hydra_params
from src.model import NERModel
from src.model.architectures import configure_optimizer
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
) -> NERModel:
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
                    
    return model


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


@hydra.main(config_path="conf", config_name="train", version_base=hydra.__version__)
def main(cfg: Any) -> None:
    """
    Main function that train the model
    """
    extra_params = {"torch_version": torch.__version__, "transformers_version": transformers.__version__}

    save_model: bool = True

    # Setup MLflow
    mlflow.set_tracking_uri("./mlflow_runs/")
    mlflow.set_experiment(experiment_name=cfg.experiment_name)

    # Set Tensorboard saving directory
    # By default hydra will log everything in output/date/time
    # This is not handy when we want to compare several runs
    tensorboard_path = f"./tensorboard_runs/{cfg.experiment_name}/"

    if cfg.test_mode:
        save_model = False
        cfg.epochs = 2
        cfg.data["split"] = ["train[:100]", "validation[:100]"]

    # Load train data
    data_train, data_valid = hydra.utils.instantiate(cfg.data, _convert_="all")
    num_labels: int = pd.Series(data_train["ner_tags"]).explode().nunique()

    # Instantiate tokenizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer, _convert_="all")

    # Construct Dataset
    train_dataset = NERDataset(
        texts=data_train["tokens"],
        labels=data_train["ner_tags"],
        tokenizer=tokenizer,
        max_len=cfg.seq_max,
        loss_ignore_index=cfg.loss_ignore_index,
        propagate_label_to_word_pieces=cfg.propagate_label_to_word_pieces,
    )
    valid_dataset = NERDataset(
        texts=data_valid["tokens"],
        labels=data_valid["ner_tags"],
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

    # Instantiate model
    model = hydra.utils.instantiate(cfg.model, _convert_="all")
    model.instantiate_network(
        num_labels=num_labels,
    )

    # Configure optimizer
    model.optimizer, model.scheduler = configure_optimizer(
        model=model,
        weight_decay=cfg.weight_decay,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        num_train_steps=int(len(data_train["tokens"]) / cfg.batch_size * cfg.epochs),
    )

    # Train & Evaluate model
    with mlflow.start_run(run_name=cfg.experiment_name), SummaryWriter(tensorboard_path) as tensorboard_writer:
        log_hydra_params(cfg)
        mlflow.log_params(extra_params)

        trained_model = train_model(
            model,
            train_data_loader=train_data_loader,
            valid_data_loader=valid_data_loader,
            epochs=cfg.epochs,
            num_labels=num_labels,
            loss_fn=hydra.utils.instantiate(cfg.loss_fn, _convert_="all"),
            tensorboard_writer=tensorboard_writer,
        )

        # Log torch model
        mlflow.pytorch.log_model(artifact_path="model", pytorch_model=trained_model)

        # Log tensorboard events
        mlflow.log_artifacts(tensorboard_path, artifact_path="tensorboard_logs")

        if save_model:
            trained_model.save(cfg.model_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
