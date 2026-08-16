import shap
import matplotlib.pyplot as plt
import pandas as pd


class ModelExplainer:
    """Classe para explicar predições de modelos de Machine Learning utilizando SHAP (SHapley Additive exPlanations)."""

    def __init__(self, model, X_train):
        """:param model: Modelo já treinado ou Pipeline.

        :param X_train: DataFrame usado no treinamento (necessário para
        explainer de background ou TreeExplainer).
        """
        self.model = model
        self.X_train = X_train

        # Se for Pipeline, extrai o modelo final
        if hasattr(self.model, "named_steps"):
            self.model_step = self.model.steps[-1][1]
        else:
            self.model_step = self.model

        # Instancia o Explainer apropriado
        if hasattr(self.model_step, "tree_") or "XGB" in type(self.model_step).__name__:
            self.explainer = shap.TreeExplainer(self.model_step)
        else:
            # Para Regressão Logística ou outros modelos
            self.explainer = shap.Explainer(self.model_step, self.X_train)

    def explain_global(self, X_test, max_display=10, plot_type="dot"):
        """Gera o gráfico de explicação global das variáveis (Beeswarm / Summary Plot).

        :param X_test: DataFrame de teste transformado ou bruto (caso use pipeline).
        :param max_display: Número máximo de variáveis exibidas no gráfico.
        :param plot_type: 'dot' (beeswarm), 'bar' (importância média).
        """
        # Se for Pipeline, transforma os dados de teste com os passos anteriores
        if hasattr(self.model, "named_steps"):
            preprocessor = self.model[:-1]
            X_test_transformed = pd.DataFrame(
                preprocessor.transform(X_test), columns=X_test.columns
            )
        else:
            X_test_transformed = X_test

        shap_values = self.explainer(X_test_transformed)

        plt.figure(figsize=(10, 6))
        if plot_type == "bar":
            shap.plots.bar(shap_values, max_display=max_display)
        else:
            shap.plots.beeswarm(shap_values, max_display=max_display)

    def explain_local(self, X_test, sample_index=0):
        """Explica uma única predição/transação (Gráfico Waterfall).

        :param X_test: DataFrame de teste.
        :param sample_index: Índice da linha/transação que deseja explicar.
        """
        if hasattr(self.model, "named_steps"):
            preprocessor = self.model[:-1]
            X_test_transformed = pd.DataFrame(
                preprocessor.transform(X_test), columns=X_test.columns
            )
        else:
            X_test_transformed = X_test

        shap_values = self.explainer(X_test_transformed)

        plt.figure(figsize=(10, 6))
        # Exibe o gráfico em cachoeira (Waterfall) para a instância escolhida
        shap.plots.waterfall(shap_values[sample_index])