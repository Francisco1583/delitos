import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, roc_curve, roc_auc_score
)
import matplotlib.pyplot as plt
import streamlit as st

# ==========================
# Funciones de Regresión Lineal Manual
# ==========================
def gradienteDescendente(X, y, theta, alpha, iteraciones):
    theta = np.array(theta).reshape(-1,1)
    costos = []
    for i in range(iteraciones):
        xtheta = X @ theta
        e = xtheta - y
        gradiente = (1/len(X) * X.T @ e)
        theta = theta - alpha * gradiente
        costo = np.mean(e**2)
        costos.append(costo)
    return theta, costos

def calculaCosto(X, y, theta):
    xtheta = X @ theta
    e = xtheta - y
    costo = np.mean(e**2)
    return costo

def buscar_mejor_alpha_iter(X_train, y_train, X_val, y_val, alphas, iteraciones_list):
    mejor_alpha = None
    mejor_iter = None
    menor_costo = float("inf")
    mejor_theta = None

    for alpha in alphas:
        for it in iteraciones_list:
            theta_ini = np.zeros((X_train.shape[1],1))
            theta_temp, _ = gradienteDescendente(X_train, y_train, theta_ini, alpha, it)
            costo_val = calculaCosto(X_val, y_val, theta_temp)

            if costo_val < menor_costo:
                menor_costo = costo_val
                mejor_alpha = alpha
                mejor_iter = it
                mejor_theta = theta_temp

    return mejor_alpha, mejor_iter, menor_costo, mejor_theta

# ==========================
# Preprocesamiento del Dataset
# ==========================
df = pd.read_csv("delitos_legible.csv")

# Mapear meses a numéricos y valores cíclicos
map_meses = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
    "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
    "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
}
df["mes_num"] = df["mes"].map(map_meses)
df["mes_sin"] = np.sin(2 * np.pi * df["mes_num"] / 12)
df["mes_cos"] = np.cos(2 * np.pi * df["mes_num"] / 12)

# ==========================
# Dataset CLASIFICACIÓN (0/1)
# ==========================
df["asalto"] = df["asalto"].map({"No": 0, "Sí": 1})
df_encoded_clasif = pd.get_dummies(df, columns=["colonia", "dia_semana", "turno"], drop_first=False)
df_final_clasif = df_encoded_clasif.drop(columns=["mes", "mes_num"])
X_clasif = df_final_clasif.drop(columns=["asalto"]).values
y_clasif = df_final_clasif["asalto"].values.reshape(-1,1)

# ==========================
# Dataset REGRESIÓN (conteo real)
# ==========================
df_reg = (
    df.groupby(["colonia","mes","dia_semana","turno","mes_num","mes_sin","mes_cos"])
      .size()
      .reset_index(name="conteo")
)
df_encoded_reg = pd.get_dummies(df_reg, columns=["colonia", "dia_semana", "turno"], drop_first=False)

# Alinear columnas con X_clasif
features_cols = df_final_clasif.drop(columns=["asalto"]).columns
df_encoded_reg = df_encoded_reg.reindex(columns=list(features_cols)+["conteo"], fill_value=0)

X_reg = df_encoded_reg.drop(columns=["conteo"]).values
y_reg = df_encoded_reg["conteo"].values.reshape(-1,1)

# ==========================
# División de datos
# ==========================
# Clasificación
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clasif, y_clasif, test_size=0.2, random_state=42, stratify=y_clasif
)

# Regresión
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Normalización segura para regresión
mean = X_train_r.mean(axis=0)
std = X_train_r.std(axis=0)
std[std == 0] = 1

X_train_norm = (X_train_r - mean) / std
X_test_norm  = (X_test_r - mean) / std

# Añadir columna de bias
X_train_bias = np.hstack([X_train_norm, np.ones((X_train_norm.shape[0],1))])
X_test_bias  = np.hstack([X_test_norm,  np.ones((X_test_norm.shape[0],1))])

# ==========================
# Regresión Lineal Manual
# ==========================
alphas = [0.001, 0.01, 0.1, 0.5, 1]
iteraciones_list = [500, 1000, 1500, 2000]

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train_bias, y_train_r, test_size=0.25, random_state=42
)

mejor_alpha, mejor_iter, menor_costo, mejor_theta = buscar_mejor_alpha_iter(
    X_train_sub, y_train_sub, X_val, y_val, alphas, iteraciones_list
)

theta_final, costos = gradienteDescendente(
    X_train_bias, y_train_r, np.zeros((X_train_bias.shape[1],1)), mejor_alpha, mejor_iter
)

# ==========================
# GridSearchCV para Regresión Logística
# ==========================
param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"]
}

grid = GridSearchCV(
    LogisticRegression(max_iter=2000),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train_c, y_train_c.ravel())
best_log_reg = grid.best_estimator_

#k-fold mejores parametros
scores = cross_validate(
    best_log_reg,
    X_clasif, y_clasif.ravel(),
    cv=5,
    scoring=["accuracy", "precision", "recall", "f1"]
)

print("\n--- Cross-Validation con mejores parámetros ---")
print("Accuracy promedio:", scores["test_accuracy"].mean())
print("Precision promedio:", scores["test_precision"].mean())
print("Recall promedio:", scores["test_recall"].mean())
print("F1 promedio:", scores["test_f1"].mean())

# ==========================
# INTERFAZ STREAMLIT
# ==========================
st.title(" Predicción de Asaltos y Conteo Estimado")

colonia = st.selectbox("Colonia:", sorted(df["colonia"].unique()))
mes = st.selectbox("Mes:", list(map_meses.keys()))
dia = st.selectbox("Día de la semana:", sorted(df["dia_semana"].unique()))
turno = st.selectbox("Turno:", sorted(df["turno"].unique()))

if st.button("Predecir"):
    #construir vector de entrada con las mismas columnas que el modelo ----
    df_in = pd.DataFrame(np.zeros((1, len(features_cols))), columns=features_cols)

    # mes -> sin/cos
    mes_num = map_meses[mes]
    df_in.at[0, "mes_sin"] = np.sin(2 * np.pi * mes_num / 12)
    df_in.at[0, "mes_cos"] = np.cos(2 * np.pi * mes_num / 12)

    # activar dummies correspondientes (si existen)
    col_col = f"colonia_{colonia}"
    dia_col = f"dia_semana_{dia}"
    tur_col = f"turno_{turno}"

    for c in [col_col, dia_col, tur_col]:
        if c in df_in.columns:
            df_in.at[0, c] = 1
        else:
            st.warning(f"La columna '{c}' no existe en el modelo. Revisa nombres/codificación.")

    # ---- FIX 2: asegurar orden exacto de columnas (por seguridad) ----
    df_in = df_in.reindex(columns=features_cols, fill_value=0)

    # Clasificación
    prob = best_log_reg.predict_proba(df_in.values)[:,1][0]
    pred = 1 if prob >= 0.5 else 0

    if pred == 0:
        st.error(f"No se espera asalto (Probabilidad: {prob:.2f})")
    else:
        st.success(f"Se espera un asalto (Probabilidad: {prob:.2f})")

        # Para REGRESIÓN lineal: normalizar con estadísticos del set de regresión
        x_norm = (df_in.values - mean) / std
        x_bias = np.hstack([x_norm, np.ones((x_norm.shape[0],1))])
        pred_reg = x_bias @ theta_final
        st.info(f"Conteo estimado de delitos: {pred_reg[0,0]:.2f}")

