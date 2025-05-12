
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# --------- Dados estruturados para os 19 subníveis ---------
dados = {
    "Subniveis": ["8E", "8D", "8C", "8B", "8A", "7B", "7A", "6B", "6A", "5A",
                  "4B", "4A", "3A", "2B", "2A", "1D", "1C", "1B", "1A"],
    "Espessura": [3.5, 5.5, 5.5, 3.0, 4.5, 4.5, 6.5, 7.5, 8.0, 8.0,
                  7.0, 5.5, 6.5, 6.5, 5.5, 6.0, 4.0, 1.5, 2.5],
    "N. AMOSTRAS": [7, 9, 8, 17, 20, 3, 13, 15, 4, 11,
                    52, 25, 14, 25, 8, 14, 7, 10, 8],
    "MOA-NF": [29, 27, 62, 54, 70, 64, 71, 78, 74, 62,
               75, 89, 84, 89, 79, 72, 71, 68, 69],
    "OP-AL": [4, 10, 3, 7, 3, 1, 2, 4, 3, 4,
              2, 1, 0, 1, 0, 1, 2, 2, 3],
    "OP-EQUI": [2, 3, 1, 4, 1, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 0],
    "2010": [6, 12, 12, 5, 9, 92, 30, 15, 18, 51,
             125, 75, 57, 44, 20, 20, 6, 0, 0],
    "2010.1": [10, 25, 25, 8, 18, 100, 71, 50, 54, 166,
               150, 91, 113, 52, 40, 64, 45, 10, 0],
    "Fe2O3": [None, None, None, None, None, None, 7.06, None, 6.66, 6.41,
              None, 5.23, 7.41, 5.99, 6.89, 5.78, 5.25, 5.52, 4.94],
    "U/Th": [None, None, None, None, None, None, 2.06, None, 1.98, 1.82,
             None, 1.81, 1.66, 1.35, 1.28, 1.11, 1.17, 1.51, 1.48],
    "Al2O3": [None, None, None, None, None, None, 14.13, None, 14.31, 13.69,
              None, 14.10, 13.62, 15.01, 14.65, 13.95, 14.89, 13.38, 13.12],
    "TiO2": [None, None, None, None, None, None, 0.57, None, 0.55, 0.59,
             None, 0.61, 0.60, 0.61, 0.61, 0.62, 0.64, 0.61, 0.63],
    "MOA": [None, None, None, None, None, None, 327, None, 286, 248,
            None, 258, 289, 286, 338, 321, 266, 251, 190],
    "TS": [None, None, None, None, None, None, 1.49, None, 1.96, 2.73,
           None, 4.54, 1.71, 1.24, 1.72, 3.98, 3.67, 1.73, 2.62],
    "TOC": [None, None, None, None, None, None, 11.93, None, 12.11, 11.70,
            None, 12.01, 13.05, 8.79, 7.69, 5.73, 7.42, 12.41, 12.97],
    "TN": [None, None, None, None, None, None, 0.43, None, 0.46, 0.44,
           None, 0.43, 0.54, 0.32, 0.29, 0.18, 0.26, 0.52, 0.54]
}

df = pd.DataFrame(dados)

colunas_y = ["Fe2O3", "U/Th", "Al2O3", "TiO2", "MOA", "TS", "TOC", "TN"]
colunas_X = ["Espessura", "N. AMOSTRAS", "MOA-NF", "OP-AL", "OP-EQUI", "2010", "2010.1"]

dados_completos = df.dropna(subset=colunas_y)
dados_faltantes = df[df[colunas_y].isnull().any(axis=1)]

X_train = dados_completos[colunas_X]
y_train = dados_completos[colunas_y]
X_pred = dados_faltantes[colunas_X]

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_pred_imputed = imputer.transform(X_pred)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_imputed, y_train)

y_pred = model.predict(X_pred_imputed)
df.loc[dados_faltantes.index, colunas_y] = y_pred

print(df.loc[:, ["Subniveis"] + colunas_y])
