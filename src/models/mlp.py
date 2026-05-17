"""Arquitectura MLP para predicción de diagnóstico de cáncer.

Implementa la función build_model() parametrizable siguiendo las restricciones
del enunciado: mínimo 3 capas ocultas Dense con BatchNormalization y Dropout,
ReLU en ocultas, Sigmoid en salida.

Referencia de parámetros objetivo: ~46.913 con input_dim=30.
"""

from __future__ import annotations

import logging
from typing import Optional

LOGGER = logging.getLogger(__name__)


def build_model(
    input_dim: int,
    hidden_units: tuple[int, ...] = (300, 100, 30),
    dropout_rates: tuple[float, ...] = (0.25, 0.25, 0.20),
    learning_rate: float = 1e-3,
) -> "tf.keras.Model":  # noqa: F821 — import diferido
    """Construye la MLP densa para clasificación binaria.

    Arquitectura:
    - 3 capas ocultas Dense con He-normal initialization.
    - BatchNormalization + Activation('relu') + Dropout en cada oculta.
    - 1 neurona de salida con activación sigmoid.

    Los defaults (300, 100, 30) están calibrados para input_dim=39
    (10 numéricas + 14 binarias + 15 de OHE sobre 6 categóricas con drop=first)
    y producen ~46.881 parámetros (objetivo del enunciado: ~46.913).

    La arquitectura de referencia del enunciado (128, 64, 32) asume input_dim≈100+.
    Con input_dim=39, re-escalamos la primera capa a 300 para mantener el orden
    de magnitud de parámetros solicitado.

    Args:
        input_dim: Número de features de entrada (tras preprocesado).
        hidden_units: Unidades por capa oculta. Mínimo 3 capas.
        dropout_rates: Tasa de dropout por capa oculta (alineada con hidden_units).
        learning_rate: Tasa de aprendizaje inicial para Adam.

    Returns:
        Modelo Keras compilado con binary_crossentropy, Adam y métricas
        de AUC-ROC, Precision y Recall.

    Raises:
        ValueError: Si hidden_units y dropout_rates tienen longitudes distintas
                    o si hay menos de 3 capas ocultas.
    """
    import tensorflow as tf  # import diferido para no bloquear si TF no está instalado

    if len(hidden_units) != len(dropout_rates):
        raise ValueError(
            f"hidden_units ({len(hidden_units)}) y dropout_rates "
            f"({len(dropout_rates)}) deben tener la misma longitud."
        )
    if len(hidden_units) < 3:
        raise ValueError("Se requieren al menos 3 capas ocultas (enunciado).")

    inputs = tf.keras.Input(shape=(input_dim,), name="input")

    x = inputs
    for i, (units, dropout) in enumerate(zip(hidden_units, dropout_rates)):
        x = tf.keras.layers.Dense(
            units,
            kernel_initializer="he_normal",
            name=f"dense_{i + 1}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn_{i + 1}")(x)
        x = tf.keras.layers.Activation("relu", name=f"relu_{i + 1}")(x)
        x = tf.keras.layers.Dropout(dropout, name=f"dropout_{i + 1}")(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs, outputs, name="mlp_cancer")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    total_params = model.count_params()
    LOGGER.info(
        "Modelo construido: input_dim=%d, capas=%s, dropout=%s, params=%d",
        input_dim,
        hidden_units,
        dropout_rates,
        total_params,
    )

    # Aviso si el número de parámetros se aleja mucho del objetivo (~46.913)
    target_params = 46_913
    ratio = total_params / target_params
    if not (0.5 <= ratio <= 2.0):
        LOGGER.warning(
            "Número de parámetros (%d) se aleja del objetivo (~46.913). "
            "Considera ajustar hidden_units.",
            total_params,
        )

    return model
