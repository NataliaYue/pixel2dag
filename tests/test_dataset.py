import numpy as np

from pixel2dag.dataset import CONJUNTOS, generar_dataset, guardar, cargar
from pixel2dag.scm import simular_paciente, MUNDO_INVERTIDO


def test_forma_y_tipo():
    X, pacientes = generar_dataset(range(5))
    assert X.shape == (5, 64, 64, 3)
    assert X.dtype == np.float32
    assert pacientes[3] == simular_paciente(3)


def test_el_mundo_se_aplica():
    _, pacientes = generar_dataset(range(10), mundo=MUNDO_INVERTIDO)
    assert pacientes == [simular_paciente(s, mundo=MUNDO_INVERTIDO) for s in range(10)]


def test_guardar_y_cargar(tmp_path):  # ← pytest pasa una carpeta temporal
    X, pacientes = generar_dataset(range(5))
    guardar("prueba", X, pacientes, carpeta=tmp_path)
    X2, tabla = cargar("prueba", carpeta=tmp_path)
    assert X2.shape == X.shape and X2.dtype == np.float32
    assert np.abs(X2 - X).max() <= 1 / 255
    assert tabla["gravedad"].tolist() == [p["gravedad"] for p in pacientes]


def test_conjuntos_no_comparten_pacientes():
    # Si un paciente estuviera en entrenamiento y en test, la evaluación sería tramposa
    vistos = set()
    for semillas, _ in CONJUNTOS.values():
        assert vistos.isdisjoint(semillas)
        vistos |= set(semillas)
