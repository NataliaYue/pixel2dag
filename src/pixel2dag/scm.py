"""Modelo causal estructural (SCM): el DAG de pacientes de pixel2dag."""

import random, math

MUNDO_ENTRENAMIENTO = {"base": -1.0, "coef_hospital_A": 1.5}
MUNDO_SIN_CONFUSION = {"base": -1.0, "coef_hospital_A": 0.0}
MUNDO_INVERTIDO = {"base": 0.5, "coef_hospital_A": -1.5}


# ---------- notebook 01 ----------
def dado(semilla, variable):
    return random.Random(f"{semilla}-{variable}")


def lanzar_monedas(rng, n, p):
    return sum(rng.random() < p for _ in range(n))


def sigmoide(x):
    return 1 / (1 + math.exp(-x))


def prob_gravedad(edad, hospital, mundo=MUNDO_ENTRENAMIENTO):
    return sigmoide(
        mundo["base"]
        + 0.05 * (edad - 60)
        + mundo["coef_hospital_A"] * (hospital == "A")
    )


PROB_MA = [0.0, 0.075, 0.2, 0.375, 0.625]
PROB_EX = [0.0, 0.0, 0.1, 0.25, 0.5]


def simular_paciente(semilla, do=None, mundo=MUNDO_ENTRENAMIENTO):
    do = do or {}
    hospital = (
        do["hospital"]
        if "hospital" in do
        else ("A" if dado(semilla, "hospital").random() < 0.5 else "B")
    )
    edad = do["edad"] if "edad" in do else dado(semilla, "edad").gauss(60, 10)
    gravedad = (
        do["gravedad"]
        if "gravedad" in do
        else lanzar_monedas(
            dado(semilla, "gravedad"), 4, prob_gravedad(edad, hospital, mundo)
        )
    )
    n_ma = lanzar_monedas(dado(semilla, "ma"), 40, PROB_MA[gravedad])
    n_ex = lanzar_monedas(dado(semilla, "ex"), 20, PROB_EX[gravedad])
    return {
        "semilla": semilla,
        "hospital": hospital,
        "edad": edad,
        "gravedad": gravedad,
        "n_ma": n_ma,
        "n_ex": n_ex,
        "referable": int(gravedad >= 2),
    }
