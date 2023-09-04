import logging
import pandas as pd
from zenml import step
from src.data_cleaning import (
    DataCleaning,
    DataDividingStrategy,
    DataPreProcessStrategy
)
from typing import Tuple
from typing_extensions import Annotated


# input and output both are a dataframe
@step
def clean_data(data: pd.DataFrame
               ) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    try:
        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        divide_strategy = DataDividingStrategy()
        data_division = DataCleaning(preprocessed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_division.handle_data()
        logging.info("Data cleaning completed")
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logging.error(e)
        raise e
        
