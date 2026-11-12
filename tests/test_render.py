import numpy as np

from pixel2dag.scm import simular_paciente
from pixel2dag.render import (
    YY,
    XX,
    MASCARA,
    pintar_mancha,
    fondo_de_ojo,
    dibujar_microaneurismas,
    renderizar,
)


def pintar_mancha_lenta(img, cy, cx, radio, color, alpha=1.0):
    # La versión original del notebook 02, solo para comparar
    w = alpha * np.exp(-((YY - cy) ** 2 + (XX - cx) ** 2) / (2 * radio**2))
    return (1 - w[..., None]) * img + w[..., None] * np.array(color)


def test_forma_y_rango():
    # Una imagen RGB de 64x64 con valores válidos
    img = renderizar(simular_paciente(17))
    assert img.shape == (64, 64, 3)
    assert img.min() >= 0 and img.max() <= 1


def test_misma_semilla_misma_imagen():
    # Reproducibilidad: la semilla es el paciente
    assert np.array_equal(
        renderizar(simular_paciente(17)), renderizar(simular_paciente(17))
    )


def test_mancha_rapida_igual_que_lenta():
    # La optimización del notebook 03 no cambia el resultado (incluidas manchas en bordes y esquinas)
    base = fondo_de_ojo(3)
    for cy, cx, r in [(32, 32, 1), (10, 50, 2.5), (0, 0, 1.5), (63, 20, 4)]:
        a = pintar_mancha_lenta(base, cy, cx, r, [1, 1, 0], 0.9)
        b = pintar_mancha(base, cy, cx, r, [1, 1, 0], 0.9)
        assert np.abs(a - b).max() < 0.02


def test_pintar_mancha_no_modifica_original():
    # Las funciones devuelven imágenes nuevas, no tocan la de entrada
    base = fondo_de_ojo(3)
    copia = base.copy()
    pintar_mancha(base, 32, 32, 3, [1, 1, 1])
    assert np.array_equal(base, copia)


def test_marca_solo_en_hospital_A():
    # La flecha hospital → imagen: la marca del escáner A
    a = renderizar(simular_paciente(17, do={"hospital": "A"}))
    b = renderizar(simular_paciente(17, do={"hospital": "B"}))
    assert a[2:7, 2:7].mean() > 0.99
    assert b[2:7, 2:7].mean() < 0.3


def test_hospital_B_mas_oscuro():
    # La otra huella del escáner: B oscurece la imagen
    a = renderizar(simular_paciente(17, do={"hospital": "A", "gravedad": 0}))
    b = renderizar(simular_paciente(17, do={"hospital": "B", "gravedad": 0}))
    assert b[MASCARA].mean() < a[MASCARA].mean()


def test_cero_lesiones_no_cambia_imagen():
    f = fondo_de_ojo(17)
    assert np.array_equal(dibujar_microaneurismas(f, 17, 0), f)


def test_lesiones_no_se_mueven():
    # Coherencia contrafactual: con más lesiones, las anteriores siguen en su sitio
    f = fondo_de_ojo(17)
    l5 = dibujar_microaneurismas(f, 17, 5)
    l10 = dibujar_microaneurismas(f, 17, 10)
    donde_5 = np.abs(l5 - f).sum(-1) > 0.05  # píxeles con lesión en l5
    assert donde_5.sum() > 0
    assert (np.abs(l10 - f).sum(-1)[donde_5] > 0.05).all()


def test_contrafactual_sano_solo_cambia_lesiones():
    # do(gravedad=0) borra las lesiones, pero el fondo y el escáner quedan igual
    original = simular_paciente(17)
    sano = simular_paciente(17, do={"gravedad": 0})
    diferentes = np.abs(renderizar(original) - renderizar(sano)).sum(-1) > 0.01
    assert 0 < diferentes.mean() < 0.35
