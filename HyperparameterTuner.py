import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)


class HyperparameterTuner:
    """Classe para automação do ajuste de hiperparâmetros de modelos do Scikit-Learn ou XGBoost."""

    def __init__(
        self,
        model,
        param_grid,
        scoring="f1",
        cv_splits=5,
        search_type="grid",
        n_iter=10,
        random_state=42,
    ):
        """
        :param model: Instância do modelo ou Pipeline (sem treinar).
        :param param_grid: Dicionário com o espaço de parâmetros a testar.
        :param scoring: Métrica de avaliação para otimização (ex: 'f1', 'roc_auc', 'recall', 'precision').
        :param cv_splits: Número de divisões na validação cruzada.
        :param search_type: Tipo de busca ('grid' para GridSearchCV ou 'random' para RandomizedSearchCV).
        :param n_iter: Número de combinações a testar caso `search_type='random'`.
        """
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
        """Executa a busca pelos melhores hiperparâmetros nos dados de treino."""
        # Usa StratifiedKFold para garantir a mesma proporção de frauda/anomalia em todas as dobras
        cv = StratifiedKFold(
            n_splits=self.cv_splits, shuffle=True, random_state=self.random_state
        )

        if self.search_type == "grid":
            self.search_object = GridSearchCV(
                estimator=self.model,
                param_grid=self.param_grid,
                scoring=self.scoring,
                cv=cv,
                n_jobs=-1,  # Utiliza todos os núcleos do processador para acelerar
                verbose=1,
            )
        elif self.search_type == "random":
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
        else:
            raise ValueError("search_type deve ser 'grid' ou 'random'.")

        print(
            f"Iniciando {self.search_type.upper()} Search para otimizar métrica '{self.scoring}'..."
        )
        self.search_object.fit(X_train, y_train)
        self.best_model = self.search_object.best_estimator_

        print("\nBusca concluída!")
        print(f"Melhor Score ({self.scoring}): {self.search_object.best_score_:.4f}")
        print("Melhores Hiperparâmetros:")
        for param, value in self.search_object.best_params_.items():
            print(f"  - {param}: {value}")

        return self.best_model

    def get_results_df(self):
        """Retorna a tabela com os resultados de todas as combinações testadas."""
        if self.search_object is None:
            raise ValueError("Execute o método .fit() antes de extrair os resultados.")

        results = pd.DataFrame(self.search_object.cv_results_)
        cols_to_keep = [c for c in results.columns if c.startswith("param_")] + [
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
        ]
        return results[cols_to_keep].sort_values(by="rank_test_score").reset_index(drop=True)