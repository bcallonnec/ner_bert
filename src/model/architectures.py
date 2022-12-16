"""Module for models architecture"""
from typing import Any, Optional, Tuple

import torch
from torch import Tensor, nn
from transformers import AdamW, get_linear_schedule_with_warmup

from src.model.torch_utils import leaky_relu_initializer, relu_initializer, tanh_initializer


def predict_labels_fn_ner(outputs: Tensor) -> Tuple[Tensor, Tensor]:
    """Predict labels for NER task"""
    # example shape of output (1, 13, 128)
    outputs = torch.sigmoid(outputs).detach().cpu()
    predict_labels: Tensor = outputs.argmax(2)
    predict_scores: Tensor = outputs.max(2)
    return predict_labels, predict_scores.values


def dense_archi_ner() -> nn.Sequential:
    """Dense architecture right after pretrained transformers model."""
    return nn.Sequential(
        nn.Linear(768, 768),
        nn.Tanh(),
    )


def output_archi_ner(in_features: int, num_labels: int) -> nn.Sequential:
    """Transformers deep Learning architecture for the NER task"""
    return nn.Sequential(
        nn.Linear(in_features, num_labels),
    )


def loss_fn_ner(output: Tensor, target: Tensor, **kwargs: Any) -> Tensor:
    """Loss function specific for NER task"""
    mask: Tensor = kwargs.pop("attention_mask")
    num_labels: Optional[int] = kwargs.pop("num_labels", None)
    lfn = nn.CrossEntropyLoss()
    active_loss: Tensor = mask.view(-1) == 1
    active_logits: Tensor = output.view(-1, num_labels)
    active_labels: Tensor = torch.where(active_loss, target.view(-1), torch.tensor(lfn.ignore_index).type_as(target))
    loss: Tensor = lfn(active_logits, active_labels)
    return loss


def default_layer_initializer(network: nn.Sequential) -> nn.Sequential:
    """
    Default layer initialization for linear layers
    It take assumption that the activation function is the same in all
    the network
    """
    for layer in network:
        if isinstance(layer, nn.Tanh):
            network = network.apply(tanh_initializer)
            break
        if isinstance(layer, nn.ReLU):
            network = network.apply(relu_initializer)
            break
        if isinstance(layer, nn.LeakyReLU):
            network = network.apply(leaky_relu_initializer)
            break

    return network


def configure_optimizer(
    model: Any,
    weight_decay: float,
    learning_rate: float,
    warmup_steps: int,
    num_train_steps: int,
) -> Tuple[Any, Any]:
    """
    Prepare optimizer and schedule (linear warmup and decay)

    Parameters
    ----------
    model: nn.Module
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_train_steps,
    )
    return optimizer, scheduler
