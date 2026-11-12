import random

import pytest

from pixel2dag.scm import (
    lanzar_monedas,
    prob_gravedad,
    simular_paciente,
    MUNDO_ENTRENAMIENTO,
    MUNDO_SIN_CONFUSION,
    MUNDO_INVERTIDO,
)


def test_misma_semilla_mismo_paciente():
    # Reproducibilidad: la semilla es el paciente
    assert simular_paciente(17) == simular_paciente(17)


def test_lanzar_monedas_extremos():
    assert lanzar_monedas(random.Random(0), 50, 0.0) == 0
    assert lanzar_monedas(random.Random(0), 50, 1.0) == 50


def test_lanzar_monedas_frecuencia():
    # Binomial(10000, 0.3): media 3000
    assert 2850 < lanzar_monedas(random.Random(0), 10_000, 0.3) < 3150


def test_prob_gravedad_valores():
    # El mecanismo hospital/edad → gravedad, con los coeficientes exactos
    assert prob_gravedad(60, "B") == pytest.approx(0.269, abs=0.001)
    assert prob_gravedad(60, "A") == pytest.approx(0.622, abs=0.001)


def test_do_gravedad_cero_sin_lesiones():
    # Intervención: forzar la gravedad se propaga a sus efectos
    p = simular_paciente(17, do={"gravedad": 0})
    assert p["n_ma"] == 0 and p["n_ex"] == 0 and p["referable"] == 0


def test_intervenir_gravedad_no_cambia_sus_causas():
    # ...pero no a sus causas
    original = simular_paciente(17)
    intervenido = simular_paciente(17, do={"gravedad": 0})
    assert intervenido["edad"] == original["edad"]
    assert intervenido["hospital"] == original["hospital"]


def test_lesiones_no_bajan_al_subir_gravedad():
    # Coherencia contrafactual: el mismo paciente más grave nunca tiene menos lesiones
    for s in range(50):
        anterior = simular_paciente(s, do={"gravedad": 0})
        for g in range(1, 5):
            actual = simular_paciente(s, do={"gravedad": g})
            assert actual["n_ma"] >= anterior["n_ma"]
            assert actual["n_ex"] >= anterior["n_ex"]
            anterior = actual


def prevalencias(mundo):
    # Auxiliar (no empieza por test_, así que pytest no la ejecuta sola)
    pacientes = [simular_paciente(s, mundo=mundo) for s in range(2000)]
    prev = {}
    for h in ["A", "B"]:
        grupo = [p for p in pacientes if p["hospital"] == h]
        prev[h] = sum(p["referable"] for p in grupo) / len(grupo)
    return prev


def test_prevalencia_mundo_entrenamiento():
    prev = prevalencias(MUNDO_ENTRENAMIENTO)
    assert prev["A"] > prev["B"] + 0.3


def test_prevalencia_mundo_sin_confusion():
    prev = prevalencias(MUNDO_SIN_CONFUSION)
    assert abs(prev["A"] - prev["B"]) < 0.08


def test_prevalencia_mundo_invertido():
    prev = prevalencias(MUNDO_INVERTIDO)
    assert prev["B"] > prev["A"] + 0.3
