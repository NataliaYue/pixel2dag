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
    assert simular_paciente(17) == simular_paciente(17)
