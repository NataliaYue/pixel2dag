"""Genera los cuatro conjuntos de datos de pixel2dag."""

import argparse
import time

from pixel2dag.dataset import CONJUNTOS, generar_dataset, guardar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--carpeta",
        default="datos",
        help="dónde guardar los datos (por defecto: datos)",
    )
    args = parser.parse_args()

    for nombre, (semillas, mundo) in CONJUNTOS.items():
        t0 = time.perf_counter()
        X, pacientes = generar_dataset(semillas, mundo=mundo)
        guardar(nombre, X, pacientes, carpeta=args.carpeta)
        print(
            f"{nombre:<20} {len(pacientes):>5} pacientes  ({time.perf_counter() - t0:.1f} s)"
        )


if __name__ == "__main__":
    main()
