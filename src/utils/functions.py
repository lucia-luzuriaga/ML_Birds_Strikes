import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
#1
def add_features_per_ticker(g):
    """
    Genera indicadores técnicos para cada ticker.

    Parameters
    ----------
    g : pandas.DataFrame
        DataFrame con los datos de una empresa ordenados por fecha.

    Returns
    -------
    pandas.DataFrame
        DataFrame con nuevas variables técnicas añadidas
        (returns, lags, medias móviles, volatilidad, RSI y MACD).
    """
    # Ordena los datos por fecha dentro de cada ticker y crea una copia del dataframe
    g = g.sort_values("Date").copy()

    # Selecciona la columna de precio de cierre
    close = g["Close"]

    # Calcula el retorno porcentual entre días consecutivos
    ret = close.pct_change()

    # Guarda el retorno simple en el dataframe
    g["return"] = ret

    # Calcula el retorno logarítmico, que suele ser más estable en series financieras
    g["log_return"] = np.log(close / close.shift(1))

    # Crea lags del precio de cierre (valores de días anteriores)
    g["Close_lag1"] = close.shift(1)
    g["Close_lag2"] = close.shift(2)
    g["Close_lag3"] = close.shift(3)

    # Crea lags del retorno
    g["return_lag1"] = ret.shift(1)
    g["return_lag2"] = ret.shift(2)
    g["return_lag3"] = ret.shift(3)

    # Calcula medias móviles simples para capturar tendencias
    g["SMA_5"] = close.rolling(5).mean()
    g["SMA_10"] = close.rolling(10).mean()
    g["SMA_20"] = close.rolling(20).mean()

    # Calcula volatilidad como desviación estándar de los retornos en distintas ventanas
    g["volatility_5"] = ret.rolling(5).std()
    g["volatility_10"] = ret.rolling(10).std()
    g["volatility_20"] = ret.rolling(20).std()

    # Calcula indicadores de momentum (diferencia entre el precio actual y el de hace N días)
    g["momentum_5"] = close - close.shift(5)
    g["momentum_10"] = close - close.shift(10)

    # Ventana utilizada para el cálculo del RSI
    window = 14

    # Diferencia entre precios consecutivos
    delta = close.diff()

    # Ganancias positivas
    gain = delta.where(delta > 0, 0.0)

    # Pérdidas (valores negativos convertidos a positivos)
    loss = (-delta.where(delta < 0, 0.0))

    # Media móvil de ganancias
    avg_gain = gain.rolling(window=window).mean()

    # Media móvil de pérdidas
    avg_loss = loss.rolling(window=window).mean()

    # Relación entre ganancias y pérdidas
    rs = avg_gain / avg_loss

    # Cálculo del indicador RSI
    g["RSI"] = 100 - (100 / (1 + rs))

    # Media móvil exponencial de 12 periodos
    g["EMA_12"] = close.ewm(span=12, adjust=False).mean()

    # Media móvil exponencial de 26 periodos
    g["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # MACD: diferencia entre las dos medias exponenciales
    g["MACD"] = g["EMA_12"] - g["EMA_26"]

    # Línea de señal del MACD (media exponencial del MACD)
    g["MACD_signal"] = g["MACD"].ewm(span=9, adjust=False).mean()

    # Devuelve el dataframe con todas las nuevas variables creadas
    return g
#2
def eval_metrics(y_true, y_pred, y_proba=None):
    """
    Calcula métricas de clasificación para evaluar el modelo.

    Parameters
    ----------
    y_true : array-like
        Valores reales.
    y_pred : array-like
        Predicciones del modelo.
    y_proba : array-like, optional
        Probabilidades predichas (necesarias para ROC-AUC).

    Returns
    -------
    dict
        Diccionario con accuracy, precision, recall, f1 y auc.
    """
    out = {}

    # Calcula la accuracy (porcentaje total de predicciones correctas)
    out["accuracy"] = accuracy_score(y_true, y_pred)

    # Calcula la precisión: proporción de predicciones positivas que son correctas
    out["precision"] = precision_score(y_true, y_pred, zero_division=0)

    # Calcula el recall: proporción de positivos reales que el modelo detecta
    out["recall"] = recall_score(y_true, y_pred, zero_division=0)

    # Calcula el F1-score: media armónica entre precision y recall
    out["f1"] = f1_score(y_true, y_pred, zero_division=0)

    # Calcula el AUC-ROC si se proporcionan probabilidades; si no, devuelve NaN
    out["auc"] = roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan

    # Devuelve el diccionario con todas las métricas calculadas
    return out
#3
def evaluar_modelo(y_true, y_pred, y_proba=None, nombre_modelo="Modelo"):
    """
    Calcula y muestra las métricas de negocio para nuestro Sistema de Trading.
    """
    # 1. Accuracy (Acierto global)
    acc = accuracy_score(y_true, y_pred)
    
    # 2. Precision (Acierto cuando el modelo predice compra: Target=1)
    precision = precision_score(y_true, y_pred, zero_division=0)
    
    # 3. Recall (Cuántas subidas reales detecta el modelo)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    # 4. F1 (Equilibrio entre Precision y Recall)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Imprimimos los resultados con un formato limpio
    print(f"--- Resultados para: {nombre_modelo} ---")
    print(f"Accuracy  (Acierto global):      {acc*100:.2f}%")
    print(f"Precision (Acierto en compras):  {precision*100:.2f}%")
    print(f"Recall    (Detección de subidas):{recall*100:.2f}%")
    print(f"F1 Score  (Equilibrio):          {f1*100:.2f}%")

    # 5. ROC-AUC (Calidad de ranking si tenemos probabilidades)
    if y_proba is not None:
        auc = roc_auc_score(y_true, y_proba)
        print(f"ROC-AUC   (Ranking):             {auc:.4f}")

    # 6. Matriz de confusión (resumen de aciertos/errores)
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_true, y_pred))
    print()