# ---------- notebook 02 ----------
IMG = 64
C = (IMG - 1) / 2
R = IMG * 0.46
YY, XX = np.mgrid[0:IMG, 0:IMG]
D = np.sqrt((YY - C) ** 2 + (XX - C) ** 2)
MASCARA = D <= R
factor = 1 - 0.35 * (D / R) ** 2


def pintar_mancha(img, cy, cx, radio, color, alpha=1.0):
    w = alpha * np.exp(-((YY - cy) ** 2 + (XX - cx) ** 2) / (2 * radio**2))
    return (1 - w[..., None]) * img + w[..., None] * np.array(color)


def dibujar_vasos(img, semilla, ny, nx):
    rng = dado(semilla, "vasos")
    for _ in range(4):
        y, x = ny, nx
        angulo = rng.uniform(0, 2 * math.pi)
        grosor = 1.2
        for paso in range(60):
            angulo += rng.gauss(0, 0.18)
            y += math.sin(angulo)
            x += math.cos(angulo)
            if math.hypot(y - C, x - C) > R:
                break
            img = pintar_mancha(img, y, x, grosor, [0.55, 0.08, 0.05], 0.6)
            grosor = max(0.5, grosor * 0.99)
    return img


def fondo_de_ojo(semilla):
    lado = dado(semilla, "lado").choice([-1, 1])
    ny, nx = 32, C + lado * 16
    img = np.zeros((IMG, IMG, 3))
    img[MASCARA] = [0.8, 0.35, 0.15]
    img = img * factor[..., None]
    img = pintar_mancha(img, ny, nx, 4, [0.98, 0.85, 0.55], 0.95)
    img = dibujar_vasos(img, semilla, ny, nx)
    return img


def posicion_en_disco(rng):
    distancia = R * 0.9 * math.sqrt(rng.random())
    angulo = rng.uniform(0, 2 * math.pi)
    return C + distancia * math.sin(angulo), C + distancia * math.cos(angulo)


def dibujar_microaneurismas(img, semilla, n_ma):
    rng = dado(semilla, "pos_ma")
    for i in range(n_ma):
        y, x = posicion_en_disco(rng)
        img = pintar_mancha(img, y, x, 0.8, [0.35, 0, 0], 0.95)
    return img


def dibujar_exudados(img, semilla, n_ex):
    rng = dado(semilla, "pos_ex")
    for i in range(n_ex):
        y, x = posicion_en_disco(rng)
        radio = rng.uniform(0.9, 1.8)
        img = pintar_mancha(img, y, x, radio, [1.0, 0.92, 0.35], 0.9)
    return img


def efecto_escaner(img, hospital, semilla):
    rng_np = np.random.default_rng(dado(semilla, "ruido").randrange(2**32))
    if hospital == "A":
        img = img + rng_np.normal(0, 0.02, img.shape)
        img[2:7, 2:7] = 1.0
    else:
        img = img * 0.8 * factor[..., None]
        img = img + rng_np.normal(0, 0.05, img.shape)
    return np.clip(img, 0, 1)


def renderizar(paciente):
    s = paciente["semilla"]
    img = fondo_de_ojo(s)
    img = dibujar_microaneurismas(img, s, paciente["n_ma"])
    img = dibujar_exudados(img, s, paciente["n_ex"])
    img = efecto_escaner(img, paciente["hospital"], s)
    return img