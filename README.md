Predicción de la Dirección del Precio en el NASDAQ-100
- Descripción:

Este proyecto desarrolla un modelo de Machine Learning supervisado de clasificación para predecir la dirección diaria del precio de cierre (sube/baja) de acciones del índice NASDAQ-100.

En lugar de predecir el precio exacto, se aborda un enfoque más realista desde el punto de vista financiero: determinar si el precio de cierre del día siguiente será superior o inferior al actual.

El objetivo es evaluar si la información histórica del mercado y los indicadores técnicos contienen señal predictiva útil.

- Dataset:

Se utiliza un dataset histórico con datos diarios de las 100 empresas actuales del NASDAQ-100:

· Periodo: 2000 – 2026

· Más de 514.000 registros

· Datos ajustados por splits

· Variables OHLCV (Open, High, Low, Close, Volume)

Los datos incluyen compañías de los sectores tecnológico, consumo, salud y comunicaciones.

- Objetivo:

Construir un modelo capaz de clasificar correctamente la dirección del precio al día siguiente:

· 1 → El precio sube

· 0 → El precio baja o se mantiene

Este planteamiento permite analizar si es posible obtener ventaja predictiva en mercados financieros altamente eficientes.

- Metodología:

· Conversión y ordenación cronológica de los datos

· Construcción de la variable objetivo mediante desplazamiento temporal

· División del dataset respetando el orden temporal (80% train / 20% test)

· Ingeniería de características basada en retornos e indicadores técnicos

· Entrenamiento y comparación de distintos modelos de clasificación

· Evaluación mediante métricas como Accuracy y ROC-AUC

Se evita el uso de shuffle para prevenir fuga de información (data leakage).

- Resultados: 

Los modelos muestran una ligera mejora respecto a un baseline simple, aunque los resultados reflejan la dificultad inherente a la predicción de mercados financieros.

El análisis confirma que, aunque existe cierta señal estadística, la magnitud del efecto es limitada, lo que es coherente con la hipótesis de eficiencia del mercado.

- Ejecución:

· Clonar el repositorio

· Instalar las dependencias necesarias

· Ejecutar main.ipynb desde el inicio hasta el final

El notebook está estructurado para ejecutarse de forma secuencial sin modificaciones.

Autores:

Proyecto desarrollado por:

Alejandro Balaguer

Blanca García

Lucía Luzuriaga
