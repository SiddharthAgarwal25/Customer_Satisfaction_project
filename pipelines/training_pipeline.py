from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.evalutation import evaluate_model
from steps.model_train import model_train

@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    trained_model =  model_train(X_train, X_test, y_train, y_test)
    r2_score, rmse = evaluate_model(trained_model, X_test, y_test)

