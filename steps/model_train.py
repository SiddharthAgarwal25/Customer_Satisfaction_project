import logging
import pandas as pd
from zenml import step
from src.model_dev import (
    HyperparameterTuner,
    LightGBMModel,
    LinearRegressionModel,
    RandomForestModel,
    XGBoostModel
)
import mlflow
from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client
from .config import ModelNameConfig
experiment_tracker = Client().activate_stack.experiment_tracker

@step(experiment_tracker= experiment_tracker.name)
def model_train(    
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    
    """
    Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
    Returns:
        model: RegressorMixin
    """
    try:
        model = None
        if config.model_name == 'LinearRegression':
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error(f"Error : {e} \n in training model : {config.model_name}")
        raise e