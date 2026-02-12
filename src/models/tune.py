import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
import time

def tune_model(X, y, n_trials: int = 40, cv_splits: int = 3):
    """
    Tunes XGBoost model to find best hyperparameters (recall).

    :param X: features
    :param y: target (0/1)
    :return: dict of best params
    """

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)  # stratified CV for imbalanced churn

    def objective(trial):
        scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)  # handle imbalance

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
            "scale_pos_weight": float(scale_pos_weight),
            "tree_method": "hist",  # faster CPU training (optional)
        }

        model = XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring="recall")
        return scores.mean()

    start = time.time()
    sampler = optuna.samplers.TPESampler(seed=42)  # reproducible tuning
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    time_taken = time.time() - start

    print("Time taken:", time_taken)
    print("Best Params:", study.best_params)
    print("Best CV Recall:", study.best_value)

    return study.best_params
