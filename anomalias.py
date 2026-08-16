import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# =====================================================================
# 1. CLASSE: IMPORTÂNCIA DAS VARIÁVEIS
# =====================================================================


class FeatureImportanceAnalyzer:
    """Calcula e visualiza a importância das variáveis do modelo."""

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = list(feature_names)
        self.importance_df = None

    def fit(self):
        # Se for Pipeline, extrai o modelo do último passo
        model_step = (
            self.model.steps[-1][1]
            if hasattr(self.model, "steps")
            else self.model
        )

        if hasattr(model_step, "feature_importances_"):
            importances = model_step.feature_importances_
            label = "Importance"
        elif hasattr(model_step, "coef_"):
            importances = np.abs(model_step.coef_[0])
            label = "Absolute_Coefficient"
        else:
            raise ValueError(
                "Modelo não suportado para extração de importâncias."
            )

        self.importance_df = (
            pd.DataFrame({"Feature": self.feature_names, label: importances})
            .sort_values(by=label, ascending=False)
            .reset_index(drop=True)
        )
        return self.importance_df

    def plot(self, top_n=10, figsize=(10, 5)):
        if self.importance_df is None:
            self.fit()

        df_to_plot = self.importance_df.head(top_n)
        val_col = df_to_plot.columns[1]

        plt.figure(figsize=figsize)
        sns.barplot(
            data=df_to_plot,
            x=val_col,
            y="Feature",
            palette="viridis",
            hue="Feature",
            legend=False,
        )
        plt.title(f"Top {top_n} Variáveis Mais Importantes")
        plt.tight_layout()
        plt.show()


# =====================================================================
# 2. CLASSE: AJUSTE DE HIPERPARÂMETROS
# =====================================================================


class HyperparameterTuner:
    """Executa a busca (GridSearch ou RandomizedSearch) mantendo validação cruzada estratificada."""

    def __init__(
        self,
        model,
        param_grid,
        scoring="f1",
        cv_splits=3,
        search_type="random",
        n_iter=5,
        random_state=42,
    ):
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv_splits = cv_splits
        self.search_type = search_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.search_object = None
        self.best_model = None

    def fit(self, X_train, y_train):
        cv = StratifiedKFold(
            n_splits=self.cv_splits, shuffle=True, random_state=self.random_state
        )

        if self.search_type == "grid":
            self.search_object = GridSearchCV(
                estimator=self.model,
                param_grid=self.param_grid,
                scoring=self.scoring,
                cv=cv,
                n_jobs=-1,
                verbose=1,
            )
        else:
            self.search_object = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=self.param_grid,
                n_iter=self.n_iter,
                scoring=self.scoring,
                cv=cv,
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1,
            )

        print(
            f"\nIniciando {self.search_type.upper()} Search (Otimizando: '{self.scoring}')..."
        )
        self.search_object.fit(X_train, y_train)
        self.best_model = self.search_object.best_estimator_

        print(
            f"Melhor Score ({self.scoring}): {self.search_object.best_score_:.4f}"
        )
        print("Melhores Parâmetros:", self.search_object.best_params_)
        return self.best_model


# =====================================================================
# 3. CLASSE: EXPLICABILIDADE COM SHAP
# =====================================================================


class ModelExplainer:
    """Gera visualizações globais e locais utilizando SHAP."""

    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train

        self.model_step = (
            self.model.steps[-1][1]
            if hasattr(self.model, "steps")
            else self.model
        )

        if (
            hasattr(self.model_step, "tree_")
            or "XGB" in type(self.model_step).__name__
        ):
            self.explainer = shap.TreeExplainer(self.model_step)
        else:
            self.explainer = shap.Explainer(self.model_step, self.X_train)

    def explain_global(self, X_test, max_display=10):
        X_test_transformed = (
            pd.DataFrame(
                self.model[:-1].transform(X_test), columns=X_test.columns
            )
            if hasattr(self.model, "steps")
            else X_test
        )
        shap_values = self.explainer(X_test_transformed)

        plt.figure(figsize=(10, 5))
        shap.plots.beeswarm(shap_values, max_display=max_display)

    def explain_local(self, X_test, sample_index=0):
        X_test_transformed = (
            pd.DataFrame(
                self.model[:-1].transform(X_test), columns=X_test.columns
            )
            if hasattr(self.model, "steps")
            else X_test
        )
        shap_values = self.explainer(X_test_transformed)

        plt.figure(figsize=(10, 5))
        shap.plots.waterfall(shap_values[sample_index])


# =====================================================================
# EXECUÇÃO DO FLUXO DO SEU CÓDIGO INTEGRADOC OM AS CLASSES
# =====================================================================

# 1. Carga e Engenharia de Recursos
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

df["Amount_log"] = np.log1p(df["Amount"])
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

# Divisão dos Dados
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ---------------------------------------------------------------------
# EXECUÇÃO DA CLASSE 1: Feature Importance Analyzer (com XGBoost)
# ---------------------------------------------------------------------
print("\n--- 1. Análise de Importância das Variáveis ---")
xgb_base = XGBClassifier(
    scale_pos_weight=10, eval_metric="logloss", random_state=42
)
xgb_base.fit(X_train, y_train)

analyzer = FeatureImportanceAnalyzer(
    model=xgb_base, feature_names=X_train.columns
)
df_importancias = analyzer.fit()
print(df_importancias.head(5))
analyzer.plot(top_n=10)

# ---------------------------------------------------------------------
# EXECUÇÃO DA CLASSE 2: Hyperparameter Tuner
# ---------------------------------------------------------------------
print("\n--- 2. Ajuste de Hiperparâmetros ---")
param_grid_xgb = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.1],
    "scale_pos_weight": [1, 10],
}

tuner = HyperparameterTuner(
    model=XGBClassifier(eval_metric="logloss", random_state=42),
    param_grid=param_grid_xgb,
    scoring="f1",
    search_type="random",
    n_iter=4,  # Poucas iterações para rápida execução
)

best_xgb = tuner.fit(X_train, y_train)
y_pred_best = best_xgb.predict(X_test)
print("\nRelatório do Melhor Modelo Encontrado:")
print(classification_report(y_test, y_pred_best))

# ---------------------------------------------------------------------
# EXECUÇÃO DA CLASSE 3: Model Explainer (SHAP)
# ---------------------------------------------------------------------
print("\n--- 3. Explicabilidade do Modelo (SHAP) ---")
# Usamos uma amostra reduzida do teste para aceleração gráfica no SHAP
X_test_sample = X_test.iloc[:200]

explainer = ModelExplainer(model=best_xgb, X_train=X_train)

# Explicação Global
print("Exibindo impacto global das variáveis...")
explainer.explain_global(X_test_sample, max_display=10)

# Explicação Local (ex: primeira transação de teste)
print("Exibindo explicação local para a primeira transação...")
explainer.explain_local(X_test_sample, sample_index=0)