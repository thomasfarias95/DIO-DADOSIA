import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureImportanceAnalyzer:
    """Classe para calcular, exibir e plotar a importância das variáveis

    de modelos baseados em árvores (XGBoost, Random Forest, etc.)
    ou modelos lineares (Regressão Logística, etc.).
    """

    def __init__(self, model, feature_names):
        """:param model: Modelo já treinado (fit executado)

        :param feature_names: Lista com os nomes das colunas/features (ex:
        X_train.columns)
        """
        self.model = model
        self.feature_names = list(feature_names)
        self.importance_df = None

    def fit(self):
        """Extrai as importâncias ou coeficientes do modelo treinado."""
        # Se o modelo for uma Pipeline do Scikit-Learn, extrai o estimador final
        if hasattr(self.model, "named_steps"):
            model_step = self.model.steps[-1][1]
        else:
            model_step = self.model

        # Modelos baseados em árvores (RandomForest, XGBoost, DecisionTree)
        if hasattr(model_step, "feature_importances_"):
            importances = model_step.feature_importances_
            metric_label = "Importance"

        # Modelos lineares (LogisticRegression, Ridge, Lasso)
        elif hasattr(model_step, "coef_"):
            importances = np.abs(
                model_step.coef_[0]
            )  # Usa o valor absoluto dos coeficientes
            metric_label = "Absolute_Coefficient"
        else:
            raise ValueError(
                "O modelo fornecido não possui 'feature_importances_' nem 'coef_'."
            )

        # Cria e ordena o DataFrame
        self.importance_df = (
            pd.DataFrame(
                {"Feature": self.feature_names, metric_label: importances}
            )
            .sort_values(by=metric_label, ascending=False)
            .reset_index(drop=True)
        )

        return self.importance_df

    def plot(self, top_n=15, figsize=(10, 6)):
        """Plota um gráfico de barras das N variáveis mais importantes."""
        if self.importance_df is None:
            self.fit()

        df_to_plot = self.importance_df.head(top_n)
        val_column = df_to_plot.columns[1]

        plt.figure(figsize=figsize)
        sns.barplot(
            data=df_to_plot,
            x=val_column,
            y="Feature",
            palette="viridis",
            hue="Feature",
            legend=False,
        )
        plt.title(f"Top {top_n} Variáveis mais Importantes", fontsize=14)
        plt.xlabel(val_column, fontsize=12)
        plt.ylabel("Variável", fontsize=12)
        plt.tight_layout()
        plt.show()