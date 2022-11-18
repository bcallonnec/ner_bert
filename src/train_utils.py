"""
Utils function for training purpuses
"""
from functools import partial
from typing import Any

import mlflow
import numpy as np
import pandas as pd


def partial2(func: Any, *args: Any, **kwargs: Any) -> Any:
    """
    normal partial requires func to be passed as a positional argument.
    This is not currently supported by instantiate, this function bridges that gap
    Inspire from: https://github.com/facebookresearch/hydra/issues/1283
    """
    return partial(func, *args, **kwargs)


def save_report(report: str, path: str) -> None:
    """

    Parameters
    ----------
    report: DataFrame
        Report Classification
    path: str
        file path

    Returns
    -------
    Log report into MlFlow
    """
    with open(path, "w") as f:
        f.write(str(report))
    mlflow.log_artifact(path)


def save_confusion_matrix(cm: np.ndarray, path: str) -> None:
    """

    Parameters
    ----------
    cm: ndarray
        Confusion matrix
    path: str
        file path

    Returns
    -------
    Log confusion matrix into MlFlow
    """
    cm_df = pd.DataFrame(cm)
    with open(path, "w") as f:
        cm_df.to_string(f)
    mlflow.log_artifact(path)
