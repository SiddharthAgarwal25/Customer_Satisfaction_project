import logging
import pandas as pd
from zenml import step
from src.evaluation import (
    RMSE,
    R2Score,
    MSE
)

from typing_extensions import Annotated
from typing import Tuple
import mlflow
from sklearn.base import RegressorMixin
from zenml.client import Client
experiment_tracker = Client().activate_stack.experiment_tracker
@step(experiment_tracker = experiment_tracker.name)
def evaluate_model( model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:


    """
    Args:
        model: RegressorMixin
        x_test: pd.DataFrame
        y_test: pd.Series
    Returns:
        r2_score: float
        rmse: float
    """
    try:
        prediction = model.predict(X_test)
        mse_score = MSE().calculate_score(y_test, prediction)
        mlflow.log_metric("mse", mse_score)
        r2_score = R2Score().calculate_score(y_test, prediction)
        mlflow.log_metric("r2_score", r2_score)
        RMSE_score = RMSE().calculate_score(y_test, prediction)
        mlflow.log_metric("RMSE_score", RMSE_score)
        return r2_score,RMSE_score
    
    except Exception as e:
        logging.error(e)
        raise e

