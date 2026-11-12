"""Generación, guardado y carga de los conjuntos de datos."""

import os

import numpy as np
import pandas as pd

from pixel2dag.scm import (
    MUNDO_ENTRENAMIENTO,
    MUNDO_SIN_CONFUSION,
    MUNDO_INVERTIDO,
    simular_paciente,
)
from pixel2dag.render import renderizar

CONJUNTOS = {  # nombre: (semillas, mundo)
    "entrenamiento": (range(0, 3000), MUNDO_ENTRENAMIENTO),
    "test_iid": (range(100000, 101000), MUNDO_ENTRENAMIENTO),
    "test_sin_confusion": (range(200000, 201000), MUNDO_SIN_CONFUSION),
    "test_invertido": (range(300000, 301000), MUNDO_INVERTIDO),
}


def generar_dataset(semillas, do=None, mundo=MUNDO_ENTRENAMIENTO):
    pacientes = [simular_paciente(s, do, mundo) for s in semillas]
    X = np.stack([renderizar(p) for p in pacientes]).astype(np.float32)
    return X, pacientes


def guardar(nombre, X, pacientes, carpeta="datos"):
    os.makedirs(carpeta, exist_ok=True)
    X8 = (X * 255).round().astype(np.uint8)
    np.savez_compressed(f"{carpeta}/{nombre}.npz", X=X8)
    pd.DataFrame(pacientes).to_csv(f"{carpeta}/{nombre}.csv", index=False)


def cargar(nombre, carpeta="datos"):
    X = np.load(f"{carpeta}/{nombre}.npz")["X"].astype(np.float32) / 255
    tabla = pd.read_csv(f"{carpeta}/{nombre}.csv")
    return X, tabla


def cargar_o_generar(nombre, carpeta="datos"):
    # Si el conjunto ya está en disco, lo carga; si no, lo genera, lo guarda y lo carga.
    if not os.path.exists(f"{carpeta}/{nombre}.npz"):
        print(f"  generando {nombre}...")
        semillas, mundo = CONJUNTOS[nombre]
        guardar(nombre, *generar_dataset(semillas, mundo=mundo), carpeta=carpeta)
    return cargar(nombre, carpeta)
