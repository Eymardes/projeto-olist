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

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

Execute os notebooks na ordem:

```
1. feature_olist.ipynb
2. eda_olist.ipynb
3. preprocessamento_olist.ipynb
4. modelo_olist.ipynb
5. tuning_olist.ipynb
```

---

## Tecnologias

Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Seaborn · Jupyter
