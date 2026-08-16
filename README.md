projeto-detectar-anomalias/
│
├── data/                       # (Opcional) Dados locais ou README de dados
├── src/                        # Código-fonte modularizado
│   ├── __init__.py             # Arquivo vazio que indica que src é um pacote
│   ├── feature_importance.py   # Classe FeatureImportanceAnalyzer
│   ├── hyperparameter_tuner.py # Classe HyperparameterTuner
│   └── model_explainer.py      # Classe ModelExplainer
│
├── .gitignore                  # Arquivos a serem ignorados pelo Git (ex: __pycache__, .env)
├── main.py                     # Script principal que orquestra e roda o projeto
├── README.md                   # Documentação detalhada do projeto
└── requirements.txt            # Dependências das bibliotecas Python
