# Olist — Previsão de Tempo de Entrega

Projeto de Machine Learning para prever o número de dias até a entrega de um pedido do e-commerce Olist.

**Target:** `dias_entrega` | **Tipo:** Regressão | **Dataset:** 96.412 pedidos

---

## Estrutura

```
olist/
├── feature_olist.ipynb          # Engenharia de features — gera dataset_features.csv
├── eda_olist.ipynb              # Análise Exploratória de Dados
├── preprocessamento_olist.ipynb # Limpeza, encoding, normalização, train/test split
├── modelo_olist.ipynb           # Treinamento: Ridge, Random Forest, XGBoost
└── tuning_olist.ipynb           # Ajuste de hiperparâmetros (GridSearch + RandomizedSearch)
```

---

## Dataset

**Fonte:** [Brazilian E-Commerce Public Dataset — Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

Os arquivos CSV já estão disponíveis na pasta `archive/` do repositório.

---

## Como executar

Crie e ative um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências do projeto:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Execute os notebooks na ordem abaixo:

```
1. feature_olist.ipynb
2. eda_olist.ipynb
3. preprocessamento_olist.ipynb
4. modelo_olist.ipynb
5. tuning_olist.ipynb
```

O notebook `preprocessamento_olist.ipynb` gera os arquivos usados no treinamento:
`X_train.csv`, `X_test.csv`, `y_train.csv` e `y_test.csv`.
Por isso, execute essa etapa antes de abrir `modelo_olist.ipynb` ou `tuning_olist.ipynb`.

O preprocessamento preserva todos os dados validos: nao remove outliers e nao aplica
winsorization/clip. Ele remove apenas linhas duplicadas, linhas com nulos/brancos,
target invalido e identificadores unicos que nao servem para treinamento.

Para executar pelo VS Code:

1. Abra a pasta do projeto no VS Code.
2. Selecione o interpretador Python da venv: `.venv/bin/python`.
3. Abra cada notebook e selecione esse mesmo kernel da venv.
4. Execute os notebooks na ordem indicada acima.

Para executar todos via terminal:

```bash
jupyter nbconvert --to notebook --execute feature_olist.ipynb --inplace
jupyter nbconvert --to notebook --execute eda_olist.ipynb --inplace
jupyter nbconvert --to notebook --execute preprocessamento_olist.ipynb --inplace
jupyter nbconvert --to notebook --execute modelo_olist.ipynb --inplace
jupyter nbconvert --to notebook --execute tuning_olist.ipynb --inplace
```

---

## Tecnologias

Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Seaborn · Jupyter
