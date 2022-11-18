"""
Define Model objects that are dependent on the pytorch library.
"""
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn import metrics
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

from src.model.architectures import default_layer_initializer, dense_archi_ner, output_archi_ner, predict_labels_fn_ner
from src.model.torch_utils import get_last_out_features


class NERModel(nn.Module):
    """
    Bert model fine tuned for ner task
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        freeze_pre_trained_layers: bool = True,
        transformers_output: str = "pooler_output",
        dense_archi: Callable = None,
        output_archi: Callable = None,
        layer_initializer: Callable = None,
        predict_labels_fn: Callable = None,
        device: str = "cpu",
    ):
        """Initialization method

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Name or path or the pretrained model loaded from hugging face
        freeze_pre_trained_layers : bool, optional
            Wether to freeze pretrained layers, by default True
        transformers_output : str, optional
            Indicate the embeddings we want from the pretrained model, by default "pooler_output"
        dense_archi : Callable, optional
            Layers architecture that receives the embeddings given by the pretrained model, by default None
        output_archi : Callable, optional
            Layers architecture right before the loss function, by default None
        layer_initializer : Callable, optional
            Callableto initializer label's weights of the model, by default None
        device : str, optional
            Indicate the device we use, by default "cpu"
        """
        super().__init__()

        # Exclude elements from save
        self.exclude_list = ["scheduler", "optimizer"]

        # Instantiate device
        self.device = device

        # Network architecture
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.pretrained_model = AutoModel.from_pretrained(pretrained_model_name_or_path, return_dict=True)
        self.dense_net = dense_archi or dense_archi_ner
        self.output_net = output_archi or output_archi_ner

        # Layer initializer
        self.layer_initializer = layer_initializer or default_layer_initializer
        self.freeze_pre_trained_layers = freeze_pre_trained_layers

        # Configure network
        self.transformers_output = transformers_output

        # Optimizer initialization
        self.optimizer: Any = None
        self.scheduler: Any = None

        # Loss function and labels weights initialization
        self.labels_weights: Optional[Sequence[float]] = None
        self._loss_fn: Optional[Callable] = None

        # Configure predict_labels function
        self.predict_labels_fn = predict_labels_fn or predict_labels_fn_ner

        # Put on device
        self.to(self.device)

    def loss_fn(self, output: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> Any:
        """Get arount mypy issue 708"""
        assert self._loss_fn is not None
        return self._loss_fn(output, target, **kwargs)

    def instantiate_network(self, num_labels: int) -> None:
        """Function called by build_network() of MelusineModel"""
        # Instantiate fine tuning network
        self.dense_net = self.dense_net()

        # Get out_features of last linear layer of dense_net
        last_linear_out_features = get_last_out_features(self.dense_net)

        # Instantiate output net which can be a Y net
        self.output_net = self.output_net(in_features=last_linear_out_features, num_labels=num_labels)

        # Initialize layers weight
        self.dense_net = self.layer_initializer(self.dense_net)
        self.output_net = self.layer_initializer(self.output_net)

        # Freeze pre trained model layers
        if self.freeze_pre_trained_layers:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

        # Put network on device
        self.to(self.device)

    def initialize_training(
        self, labels_weights: Optional[Sequence[float]] = None, loss_fn: Optional[Callable] = None
    ) -> None:
        """
        Instantiate loss function and labels weights
        Parameters
        ----------
        labels_weights: Optional[Sequence[float]]
            Weight associated with each labels
        loss_fn: Optional[Callable]
            The loss function choosen for the problem
        """
        # Configure weights for loss function
        self.labels_weights = labels_weights
        if labels_weights is not None:
            self.labels_weights = torch.FloatTensor(labels_weights).to(self.device)

        # Configure loss function
        self._loss_fn = loss_fn or predict_labels_fn_ner

    def forward(self, **inputs: Any) -> Any:
        """Apply torch forward algorithm"""
        # Get embeddings data
        output = self.pretrained_model(**inputs)

        # Dense net
        output = self.dense_net(output[self.transformers_output])

        # Output net that give prediction
        output = self.output_net(output)

        return output

    def training_step(self, batch: Any, num_labels: Optional[int] = None) -> Any:
        """Training step for pytorch lightning Trainer"""
        for key, var in batch.items():
            batch[key] = var.to(self.device)

        targets = batch.pop("targets")

        self.optimizer.zero_grad()

        outputs = self(**batch)

        loss = self.loss_fn(outputs, targets, weights=self.labels_weights, num_labels=num_labels, **batch)

        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        return loss

    def _shared_eval_step(self, batch: Any, num_labels: Optional[int] = None) -> Dict[str, Any]:
        """Common for test & validation step"""
        for key, var in batch.items():
            batch[key] = var.to(self.device)
        targets = batch.pop("targets")

        outputs = self(**batch)

        loss = self.loss_fn(outputs, targets, weights=self.labels_weights, num_labels=num_labels, **batch)

        return {"loss": loss, "outputs": outputs, "targets": targets}

    def _shared_eval_epoch_end(self, outputs: Any) -> Dict[str, float]:
        """Common for test & validation step"""
        preds = torch.cat([x["outputs"] for x in outputs]).detach().cpu()
        targets = torch.cat([x["targets"] for x in outputs]).detach().cpu()
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        # Get predicted labels based on threshold
        preds, _ = self.predict_labels_fn(preds)

        if len(preds.shape) > 1:
            preds = preds.reshape(-1)
            targets = targets.reshape(-1)

        metric_dict = {}
        metric_dict["val_loss"] = loss.item()
        metric_dict["val_acc"] = metrics.accuracy_score(targets, preds)
        metric_dict["val_f1"] = metrics.f1_score(targets, preds, average="macro")
        return metric_dict

    def validation_step(self, batch: Any, num_labels: Optional[int] = None) -> Dict[str, Any]:
        """Validation step for pytorch lightning Trainer"""
        return self._shared_eval_step(batch=batch, num_labels=num_labels)

    def test_step(self, batch: Any, num_labels: Optional[int] = None) -> Dict[str, Any]:
        """Test step"""
        return self._shared_eval_step(batch=batch, num_labels=num_labels)

    def validation_epoch_end(self, outputs: Any) -> Dict[str, float]:
        """Validation epoch end log metrics"""
        return self._shared_eval_epoch_end(outputs)

    def test_epoch_end(self, outputs: Any) -> Dict[str, Any]:
        """Test epoch end log metrics"""
        return self._shared_eval_epoch_end(outputs)

    def predict_step(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict"""
        _ = batch.pop("targets")
        for key, var in batch.items():
            batch[key] = var.to(self.device)

        outputs = self(**batch)

        # Get predicted labels
        preds, logits = self.predict_labels_fn(outputs)

        return preds, logits

    def predict(self, data_loader: DataLoader, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run predictions

        Parameters
        ----------
        data_loader: DataLoader
            Model inputs
        """
        # Turn model on eval mode
        self.eval()

        # Get prpgress_bar
        progress_bar = kwargs.pop("progress_bar", False)

        # Initialize results
        preds = torch.empty((0,))
        probas = torch.empty((0,))

        # Prediction loop
        with torch.no_grad():
            if progress_bar:
                for _, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
                    preds_batch, probas_batch = self.predict_step(batch)

                    preds = torch.cat((preds, preds_batch.detach()), dim=0).detach()
                    probas = torch.cat((probas, probas_batch.detach()), dim=0).detach()
            else:
                for _, batch in enumerate(data_loader):
                    preds_batch, probas_batch = self.predict_step(batch)

                    preds = torch.cat((preds, preds_batch.detach()), dim=0).detach()
                    probas = torch.cat((probas, probas_batch.detach()), dim=0).detach()

        # Convert results to right data type
        preds = preds.detach().cpu().to(preds_batch.dtype).numpy()
        probas = probas.detach().cpu().to(probas_batch.dtype).numpy()

        return preds, probas

    def save(self, path: str) -> None:
        """Save model at given path"""
        # Set device model to cpu
        self.device = "cpu"

        # Set training arguments we do not save to None
        for elem in self.exclude_list:
            setattr(self, elem, None)

        # Don't save pretrained model if layers are freezed
        if self.freeze_pre_trained_layers:
            self.pretrained_model = None

        # Save model
        torch.save(self, path)

    @classmethod
    def load(cls, path: str, pretrained_model_path: Optional[str] = None) -> Any:
        """
        Load the model given its path

        Parameters
        ----------
        path: str
            Load path
        pretrained_model_path: Optional[str]
            Pretrained model path
        Returns
        -------
        NERModel
        """
        loaded_model = torch.load(path, map_location=torch.device("cpu"))

        # Load pretrained model
        if (
            hasattr(loaded_model, "pretrained_model")
            and loaded_model.pretrained_model is None
            and pretrained_model_path
        ):
            loaded_pretrained_model = AutoModel.from_pretrained(pretrained_model_path, return_dict=True)
            loaded_model.pretrained_model = loaded_pretrained_model

        return loaded_model
