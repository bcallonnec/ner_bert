"""Utils methods relatives to mlflow"""

from typing import Any, Dict

import mlflow
import pandas as pd
from omegaconf import OmegaConf


def log_hydra_params(train_params: Dict[str, Any]) -> None:
    """
    Hack to save hydra params to MLflow.

    Parameters
    ----------
    train_params: DictConfig
    """
    df_params = pd.json_normalize(OmegaConf.to_container(train_params))
    params = df_params.to_dict(orient="records")[0]
    mlflow.log_params(params)
