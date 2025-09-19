import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, roc_curve, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import streamlit as st

# ==========================
# Funciones de Regresión Lineal Manual
# ==========================
def gradienteDescendente(X, y, theta, alpha, iteraciones):
    theta = np.array(theta).reshape(-1,1)  # Asegurar dimensión correcta
    costos = []  # Historial de costos
    for i in range(iteraciones):
        xtheta = X @ theta  # Predicción actual
        e = xtheta - y  # Error
        gradiente = (1/len(X) * X.T @ e)  # Gradiente promedio
        theta = theta - alpha * gradiente  # Actualización de parámetros
        costo = np.mean(e**2)  # MSE
        costos.append(costo)
    return theta, costos

def calculaCosto(X, y, theta):
    xtheta = X @ theta  # Predicción
    e = xtheta - y  # Error
    costo = np.mean(e**2)  # MSE
    return costo

def buscar_mejor_alpha_iter(X_train, y_train, X_val, y_val, alphas, iteraciones_list):
    mejor_alpha = None
    mejor_iter = None
    menor_costo = float("inf")
    mejor_theta = None

    for alpha in alphas:  # Probar cada tasa de aprendizaje
        for it in iteraciones_list:  # Probar cada número de iteraciones
            theta_ini = np.zeros((X_train.shape[1],1))  # Inicializar en ceros
            theta_temp, _ = gradienteDescendente(X_train, y_train, theta_ini, alpha, it)
            costo_val = calculaCosto(X_val, y_val, theta_temp)  # Evaluar en validación

            if costo_val < menor_costo:  # Guardar si mejora
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
df["mes_num"] = df["mes"].map(map_meses)  # Convertir mes a número
df["mes_sin"] = np.sin(2 * np.pi * df["mes_num"] / 12)  # Componente cíclica seno
df["mes_cos"] = np.cos(2 * np.pi * df["mes_num"] / 12)  # Componente cíclica coseno

# ==========================
# Dataset CLASIFICACIÓN (0/1)
# ==========================
df["asalto"] = df["asalto"].map({"No": 0, "Sí": 1})  # Binario para clasificación
df_encoded_clasif = pd.get_dummies(df, columns=["colonia", "dia_semana", "turno"], drop_first=False)  # One-hot encoding
df_final_clasif = df_encoded_clasif.drop(columns=["mes", "mes_num"])  # Quitar columnas redundantes
X_clasif = df_final_clasif.drop(columns=["asalto"]).values  # Features
y_clasif = df_final_clasif["asalto"].values.reshape(-1,1)  # Target

# ==========================
# Dataset REGRESIÓN (conteo real)
# ==========================
df_reg = (
    df.groupby(["colonia","mes","dia_semana","turno","mes_num","mes_sin","mes_cos"])
      .size()  # Contar ocurrencias
      .reset_index(name="conteo")  # Crear columna conteo
)
df_encoded_reg = pd.get_dummies(df_reg, columns=["colonia", "dia_semana", "turno"], drop_first=False)  # One-hot

# Alinear columnas con X_clasif
features_cols = df_final_clasif.drop(columns=["asalto"]).columns  # Columnas de clasificación
df_encoded_reg = df_encoded_reg.reindex(columns=list(features_cols)+["conteo"], fill_value=0)  # Alinear columnas

X_reg = df_encoded_reg.drop(columns=["conteo"]).values
y_reg = df_encoded_reg["conteo"].values.reshape(-1,1)

# ==========================
# División de datos
# ==========================
# Clasificación
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clasif, y_clasif, test_size=0.2, random_state=42, stratify=y_clasif  # Mantener proporción de clases
)

# Regresión
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Normalización segura para regresión
X_train_r = np.array(X_train_r, dtype=np.float64)  # Fix dtype para numpy
X_test_r = np.array(X_test_r, dtype=np.float64)  # Fix dtype para numpy
mean = X_train_r.mean(axis=0)  # Media por columna
std = X_train_r.std(axis=0)  # Desviación estándar
std[std == 0] = 1  # Evitar división por cero

X_train_norm = (X_train_r - mean) / std  # Normalización Z-score
X_test_norm  = (X_test_r - mean) / std  # Usar estadísticas de train

# Añadir columna de bias
X_train_bias = np.hstack([X_train_norm, np.ones((X_train_norm.shape[0],1))])  # Añadir término de sesgo
X_test_bias  = np.hstack([X_test_norm,  np.ones((X_test_norm.shape[0],1))])  # Añadir término de sesgo

# ==========================
# Regresión Lineal Manual
# ==========================
alphas = [0.001, 0.01, 0.1, 0.5, 1]  # Tasas de aprendizaje a probar
iteraciones_list = [500, 1000, 1500, 2000]  # Iteraciones a probar

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train_bias, y_train_r, test_size=0.25, random_state=42  # 25% para validación
)

mejor_alpha, mejor_iter, menor_costo, mejor_theta = buscar_mejor_alpha_iter(
    X_train_sub, y_train_sub, X_val, y_val, alphas, iteraciones_list
)

print("\n--- Búsqueda Manual de Hiperparámetros (Regresión Lineal) ---")
print(f"Mejor alpha: {mejor_alpha}, Mejor iteraciones: {mejor_iter}, Costo en validación: {menor_costo}")

theta_final, costos = gradienteDescendente(
    X_train_bias, y_train_r, np.zeros((X_train_bias.shape[1],1)), mejor_alpha, mejor_iter
)

print("\n--- Regresión Lineal Manual ---")
print("Costo en entrenamiento:", calculaCosto(X_train_bias, y_train_r, theta_final))
print("Costo en prueba:", calculaCosto(X_test_bias, y_test_r, theta_final))

y_pred_reg = X_test_bias @ theta_final
print("MSE en test:", mean_squared_error(y_test_r, y_pred_reg))

# ==========================
# GridSearchCV para Regresión Logística
# ==========================
param_grid = {
    "C": [0.01, 0.1, 1, 10],  # Regularización inversa
    "penalty": ["l1", "l2"],  # Tipo de penalización
    "solver": ["liblinear", "saga"]  # Optimizadores compatibles
}

grid = GridSearchCV(
    LogisticRegression(max_iter=2000),  # Modelo base
    param_grid,  # Combinaciones a probar
    cv=5,  # 5-fold cross validation
    scoring="f1",  # Métrica de optimización
    n_jobs=-1  # Usar todos los cores
)

grid.fit(X_train_c, y_train_c.ravel())  # Buscar mejores parámetros
best_log_reg = grid.best_estimator_  # Extraer mejor modelo

print("\n--- GridSearchCV Resultados ---")
print("Mejores parámetros:", grid.best_params_)
print("Mejor F1 Score en CV:", grid.best_score_)

#k-fold mejores parametros
scores = cross_validate(
    best_log_reg,  # Modelo con mejores hiperparámetros
    X_clasif, y_clasif.ravel(),  # Dataset completo
    cv=5,  # 5 folds
    scoring=["accuracy", "precision", "recall", "f1"]  # Múltiples métricas
)

print("\n--- Cross-Validation con mejores parámetros ---")
print("Accuracy promedio:", scores["test_accuracy"].mean())
print("Precision promedio:", scores["test_precision"].mean())
print("Recall promedio:", scores["test_recall"].mean())
print("F1 promedio:", scores["test_f1"].mean())

y_scores = best_log_reg.predict_proba(X_test_c)[:,1]  # Probabilidades clase positiva
fpr, tpr, thresholds = roc_curve(y_test_c, y_scores)  # Curva ROC
auc = roc_auc_score(y_test_c, y_scores)  # Área bajo la curva

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="blue")
plt.plot([0,1],[0,1],"--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC - Regresión Logística")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("plots/1_roc_curve.png", dpi=300, bbox_inches='tight')
plt.show()

# ==========================
# VISUALIZACIONES ADICIONALES
# ==========================

# 1. Confusion Matrix
print("\n--- Matriz de Confusión ---")
y_pred_best = best_log_reg.predict(X_test_c)  # Predicciones finales
cm = confusion_matrix(y_test_c, y_pred_best)  # Matriz de confusión
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['No Asalto', 'Asalto'])
plt.yticks(tick_marks, ['No Asalto', 'Asalto'])
plt.ylabel('Verdadero')
plt.xlabel('Predicho')

# Add text annotations
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")  # Anotar valores

plt.tight_layout()
plt.savefig("plots/2_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. Metrics Comparison
print("\n--- Comparación de Métricas ---")
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
metrics_values = [
    accuracy_score(y_test_c, y_pred_best),  # Proporción correcta
    precision_score(y_test_c, y_pred_best),  # VP/(VP+FP)
    recall_score(y_test_c, y_pred_best),  # VP/(VP+FN)
    f1_score(y_test_c, y_pred_best)  # Media armónica precision-recall
]

plt.figure(figsize=(8,5))
bars = plt.bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
plt.title('Comparación de Métricas de Clasificación')
plt.ylabel('Score')
plt.ylim(0, 1)

# Add value labels on bars
for bar, value in zip(bars, metrics_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom')  # Añadir valores sobre barras

plt.grid(axis='y', alpha=0.3)
plt.savefig("plots/3_metrics_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Crime Distribution by Turno
print("\n--- Distribución de Crímenes por Turno ---")
turno_counts = df.groupby(['turno', 'asalto']).size().unstack(fill_value=0)  # Tabla cruzada

plt.figure(figsize=(8,5))
turno_counts.plot(kind='bar', stacked=True, color=['lightcoral', 'steelblue'])
plt.title('Distribución de Asaltos por Turno')
plt.xlabel('Turno')
plt.ylabel('Cantidad de Casos')
plt.legend(['No Asalto', 'Asalto'], loc='upper right')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("plots/4_crime_by_turno.png", dpi=300, bbox_inches='tight')
plt.show()
