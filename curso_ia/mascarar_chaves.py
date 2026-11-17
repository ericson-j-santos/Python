"""Gera cópia ZIP de notebooks com tokens de API aparentes mascarados.

Não extrai nem executa arquivos. O arquivo original nunca é alterado.
Não garante detecção completa de segredos.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

PADRAO = re.compile(rb"sk-[A-Za-z0-9_-]{18,}|AIza[0-9A-Za-z_-]{25,}")
MASCARA = b"CHAVE_REMOVIDA_NAO_EXECUTAR"


def mascarar(origem: Path, destino: Path) -> dict:
    if origem.resolve() == destino.resolve():
        raise ValueError("A origem e o destino devem ser diferentes")
    if destino.exists():
        raise FileExistsError("Destino já existe; não sobrescrever")
    registro = []
    with zipfile.ZipFile(origem) as entrada, zipfile.ZipFile(destino, "x", compression=zipfile.ZIP_DEFLATED) as saida:
        nomes = entrada.namelist()
        if len(nomes) != len(set(nomes)):
            raise ValueError("ZIP contém nomes duplicados")
        for membro in entrada.infolist():
            bruto = entrada.read(membro)
            novo = bruto
            alteracoes = 0
            if membro.filename.lower().endswith(".ipynb"):
                novo, alteracoes = PADRAO.subn(MASCARA, bruto)
                if alteracoes:
                    json.loads(novo.decode("utf-8"))
                    registro.append({"arquivo": membro.filename, "substituicoes": alteracoes,
                                     "sha256_anterior": hashlib.sha256(bruto).hexdigest(),
                                     "sha256_atual": hashlib.sha256(novo).hexdigest()})
            saida.writestr(membro.filename, novo)
    with zipfile.ZipFile(destino) as teste:
        if teste.testzip() is not None:
            raise RuntimeError("Falha de integridade na cópia ZIP")
    return {"arquivos": len(nomes), "notebooks_alterados": len(registro),
            "substituicoes": sum(r["substituicoes"] for r in registro), "registro": registro}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origem", required=True, type=Path)
    parser.add_argument("--destino", required=True, type=Path)
    args = parser.parse_args()
    try:
        resultado = mascarar(args.origem, args.destino)
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as erro:
        parser.error(str(erro))
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
