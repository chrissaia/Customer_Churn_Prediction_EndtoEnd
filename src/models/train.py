import mlflow.sklearn
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, auc
import time


def train_model(df: pd.DataFrame, target_col: str):
    '''
    Trains XGBoost model and tracks performance metrics using MLFlow

    :param df: dataframe containing training data
    :param target_col:
    '''

    X = df.drop(columns=target_col)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    estimators = 300

    model = XGBClassifier(
        n_estimators=estimators,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )

    with mlflow.start_run():
        # Train model
        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        precision = precision_score(y_test, preds)
        auc_score = auc(y_test, preds)


        # Log params, metrics, and model
        mlflow.log_param("n_estimators", estimators)
        mlflow.log_metric("train_time", train_time)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", auc_score)

        mlflow.xgboost.log_model(model, "model")

        # Log dataset so it shows in MLflow UI
        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print(f"Model trained. Accuracy: {acc:.4f}, Recall: {rec:.4f}")
