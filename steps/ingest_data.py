import logging
import pandas as pd
from zenml import step


class IngestData:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)

# Function that takes a path as string and returns a data frame.
@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingest data from the path

    Args:
        data_path: path to dataset
    
    Returns:
        Pandas dataframe
    """

    try:
        ingest_data = IngestData(data_path)
        return ingest_data.get_data()
         
    except Exception as e:
        logging.error(f"Error :{e}, while ingesting the data")
        return e