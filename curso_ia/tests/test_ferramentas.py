"""Verificações positivas, negativas e de repetição das ferramentas próprias."""
from pathlib import Path
import json
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from mascarar_chaves import mascarar
from gerar_indice_ia import gerar


def test_mascarar_sem_expor_token_ou_alterar_origem(tmp_path):
    a, b = tmp_path / "original.zip", tmp_path / "seguro.zip"
    segredo = "sk-" + "A" * 35
    with zipfile.ZipFile(a, "w") as z:
        z.writestr("1.Introdução/a.ipynb", json.dumps({"nbformat": 4, "cells": [{"source": [segredo]}]}))
        z.writestr("1.Introdução/b.txt", "preservar")
    antes = a.read_bytes()
    resultado = mascarar(a, b)
    assert a.read_bytes() == antes
    assert resultado["notebooks_alterados"] == 1
    with zipfile.ZipFile(b) as z:
        assert segredo.encode() not in z.read("1.Introdução/a.ipynb")
        assert z.read("1.Introdução/b.txt") == b"preservar"
    with pytest.raises(FileExistsError):
        mascarar(a, b)


def test_rejeita_destruicao_da_origem(tmp_path):
    a = tmp_path / "mesmo.zip"
    with pytest.raises(ValueError):
        mascarar(a, a)


def test_indice_nao_espelha_conteudo_e_pesquisa_local(tmp_path):
    zip_ = tmp_path / "a.zip"
    html = tmp_path / "indice.html"
    with zipfile.ZipFile(zip_, "w") as z:
        z.writestr("1.Introdução/exemplo.ipynb", b"CONTEUDO_PRIVADO_NAO_PODE_IR_PARA_INDICE")
    r = gerar(zip_, html)
    pagina = html.read_text(encoding="utf-8")
    assert r["arquivos"] == 1 and r["modulos"] == 1
    assert "exemplo.ipynb" in pagina
    assert "CONTEUDO_PRIVADO_NAO_PODE_IR_PARA_INDICE" not in pagina
    assert 'document.createElement("a")' in pagina
    with pytest.raises(FileExistsError):
        gerar(zip_, html)
