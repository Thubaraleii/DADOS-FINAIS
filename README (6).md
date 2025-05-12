
# Predição de Teores Geoquímicos com Random Forest

Este projeto utiliza regressão por Random Forest para preencher valores ausentes em teores geoquímicos de 19 subníveis.

## Dependências

```bash
pip install pandas scikit-learn
```

## Uso

Execute o script:

```bash
python prever_teores.py
```

O script imprime um DataFrame com os valores previstos para:

- Fe2O3
- U/Th
- Al2O3
- TiO2
- MOA
- TS
- TOC
- TN

## Sobre

Os dados usados são geocientíficos, com foco em paleoambientes e geoquímica em subníveis sedimentares.
