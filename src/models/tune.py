import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import time

def tune_model(X, y):
    '''
    Tunes XGBoost model to find best hyperparameters

    :param X: All data excluding target columns
    :param y: target columns
    :return: dict of best params
    '''

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }

        model = XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring="recall")
        return scores.mean()

    start = time.time()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40)
    time_taken = time.time() - start

    print("Time taken: ", time_taken)
    print("Best Params:", study.best_params)
    return study.best_params