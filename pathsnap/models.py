"""Model zoo: candidate classifiers from several algorithm families.

Every model outputs P(candidate is the reference arc | features) via
predict_proba, so all of them can feed the HMM as emission probabilities.
Scale-sensitive models (linear, KNN, SVM, MLP) are wrapped with a
StandardScaler inside a Pipeline so the scaler is fitted on training folds only.
"""
from lightgbm import LGBMClassifier
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def make_model(name, seed=42, n_jobs=-1, **params):
    """n_jobs: threads per model (set to 1 when several models train concurrently)."""
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=params.get("C", 1.0)))
    if name == "gnb":
        return make_pipeline(StandardScaler(), GaussianNB())
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=params.get("n_neighbors", 25), weights="distance", n_jobs=n_jobs))
    if name == "dtree":
        return DecisionTreeClassifier(max_depth=params.get("max_depth", 12),
                                      min_samples_leaf=params.get("min_samples_leaf", 50), random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=params.get("n_estimators", 200),
                                      max_depth=params.get("max_depth", 16),
                                      min_samples_leaf=params.get("min_samples_leaf", 20),
                                      n_jobs=n_jobs, random_state=seed)
    if name == "extratrees":
        return ExtraTreesClassifier(n_estimators=params.get("n_estimators", 200),
                                    max_depth=params.get("max_depth", 18),
                                    min_samples_leaf=params.get("min_samples_leaf", 20),
                                    n_jobs=n_jobs, random_state=seed)
    if name == "histgb":
        return HistGradientBoostingClassifier(max_iter=params.get("max_iter", 300),
                                              learning_rate=params.get("learning_rate", 0.1),
                                              random_state=seed)
    if name == "lightgbm":
        defaults = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0)
        defaults.update(params)
        return LGBMClassifier(random_state=seed, n_jobs=n_jobs, verbose=-1, **defaults)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=params.get("hidden", (64, 32)), alpha=params.get("alpha", 1e-4),
            early_stopping=True, validation_fraction=0.1, max_iter=200, random_state=seed))
    raise ValueError(name)


MODEL_INFO = {
    "logreg": ("Linear / probabilistic", "StandardScaler + LogisticRegression C=1.0"),
    "gnb": ("Linear / probabilistic", "StandardScaler + GaussianNB"),
    "knn": ("Instance-based", "k=25, distance-weighted, scaled"),
    "dtree": ("Tree-based", "max_depth=12, min_samples_leaf=50"),
    "rf": ("Ensemble (bagging)", "200 trees, max_depth=16, min_leaf=20"),
    "extratrees": ("Ensemble (bagging)", "200 trees, max_depth=18, min_leaf=20"),
    "histgb": ("Boosting", "HistGB 300 iters, lr=0.1"),
    "lightgbm": ("Boosting", "LightGBM 400 trees, lr=0.05, 63 leaves"),
    "mlp": ("Neural network", "MLP (64,32), adam, early stopping"),
}
