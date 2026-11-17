"""Executa o exercício original Clusters.ipynb offline, sem modificar o curso.

Requer Python com numpy, scipy, scikit-learn e matplotlib já instalados.
Não instala pacotes nem envia dados a serviços externos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

CAMINHO = "3.Algoritmos de Machine Learning/Clusters.ipynb"
SHA256_ESPERADO = "32e695dd206872c0e3dba468090d32cbfcc1a6044bf6860eb5e74d3d0aa622d7"

# Processamento isolado: somente um notebook específico e certificado pode executar.
_EXECUTOR = r'''
import contextlib
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

def bloquear(*args, **kwargs):
    raise RuntimeError("Acesso à rede bloqueado durante a validação")
socket.socket.connect = bloquear
socket.socket.connect_ex = bloquear
socket.create_connection = bloquear
np.random.seed(42)
dados = json.load(sys.stdin)
ambiente = {"__name__": "__main__"}
figuras = []
with tempfile.TemporaryDirectory(prefix="curso_ia_offline_") as pasta:
    def guardar_figura(*args, **kwargs):
        caminho = Path(pasta) / ("figura_%02d.png" % (len(figuras) + 1))
        plt.savefig(caminho, format="png", dpi=70)
        if not caminho.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise AssertionError("Imagem gerada inválida")
        figuras.append(caminho.stat().st_size)
        plt.close("all")
    plt.show = guardar_figura
    with contextlib.redirect_stdout(io.StringIO()):
        for codigo in dados["codigos"]:
            exec(compile(codigo, "Clusters.ipynb", "exec"), ambiente)
    iris = ambiente["iris"]
    km = ambiente["kmeans"].labels_
    db = ambiente["dbscan_labels"]
    ag = ambiente["agglo_labels"]
    resultado = {
        "instancias": int(len(iris.data)),
        "atributos": int(iris.data.shape[1]),
        "classes_verdadeiras": int(len(set(iris.target))),
        "kmeans_grupos": int(len(set(km))),
        "dbscan_grupos_com_ruido": int(len(set(db))),
        "hierarquico_grupos": int(len(set(ag))),
        "graficos_png_validos": len(figuras),
        "tamanhos_graficos": figuras,
        "ari_kmeans": round(float(adjusted_rand_score(iris.target, km)), 6),
        "ari_hierarquico": round(float(adjusted_rand_score(iris.target, ag)), 6),
    }
print(json.dumps(resultado, ensure_ascii=False, sort_keys=True))
'''


def extrair_codigos(pacote: Path) -> list[str]:
    """Valida identidade e formato antes de considerar o código executável."""
    with zipfile.ZipFile(pacote, "r") as zip_:
        if CAMINHO not in zip_.namelist():
            raise ValueError("Exercício Clusters.ipynb não encontrado no ZIP")
        if zip_.getinfo(CAMINHO).file_size > 2_000_000:
            raise ValueError("Exercício excede o tamanho máximo permitido")
        bruto = zip_.read(CAMINHO)
    digest = hashlib.sha256(bruto).hexdigest()
    if digest != SHA256_ESPERADO:
        raise ValueError("SHA-256 do exercício diferente do conteúdo autorizado")
    dados = json.loads(bruto)
    if dados.get("nbformat") != 4 or not isinstance(dados.get("cells"), list):
        raise ValueError("Notebook Jupyter inválido")
    codigos = ["".join(c.get("source", [])) for c in dados["cells"] if c.get("cell_type") == "code"]
    if len(codigos) != 9 or any(not isinstance(x, str) for x in codigos):
        raise ValueError("Estrutura de células inesperada")
    return codigos


def executar_offline(pacote: Path, *, timeout: int = 35) -> dict:
    """Executa com limite de tempo, sem chamada externa e sem gravar no ZIP."""
    codigos = extrair_codigos(pacote)
    env = {"MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    # Mantém PATH e recursos mínimos necessários ao Python, sem herdar chaves de API.
    for campo in ("PATH", "SYSTEMROOT", "WINDIR", "HOME", "TMPDIR", "TEMP", "TMP"):
        if campo in os.environ:
            env[campo] = os.environ[campo]
    resultado = subprocess.run(
        [sys.executable, "-I", "-c", _EXECUTOR],
        input=json.dumps({"codigos": codigos}), text=True,
        capture_output=True, timeout=timeout, env=env, check=False,
    )
    if resultado.returncode != 0:
        raise RuntimeError("Exercício falhou: " + resultado.stderr[-2500:])
    dados = json.loads(resultado.stdout)
    assert dados["instancias"] == 150 and dados["atributos"] == 4
    assert dados["kmeans_grupos"] == 3 and dados["hierarquico_grupos"] == 3
    assert dados["graficos_png_validos"] == 4 and all(x > 500 for x in dados["tamanhos_graficos"])
    return {"status": "aprovado", "exercicio_sha256": SHA256_ESPERADO, **dados}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, type=Path, help="ZIP do curso, sem upload para serviços externos")
    parser.add_argument("--saida", type=Path, help="Caminho opcional para gravar evidência JSON")
    args = parser.parse_args()
    try:
        resultado = executar_offline(args.zip)
    except (OSError, zipfile.BadZipFile, ValueError, RuntimeError, AssertionError, subprocess.TimeoutExpired) as erro:
        print(json.dumps({"status": "reprovado", "erro": str(erro)}, ensure_ascii=False), file=sys.stderr)
        return 1
    texto = json.dumps(resultado, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.saida:
        args.saida.write_text(texto, encoding="utf-8")
    print(texto, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
