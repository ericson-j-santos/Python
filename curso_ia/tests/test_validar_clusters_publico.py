"""Controles de integridade que dispensam o material privado do curso."""
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from validar_clusters import CAMINHO, extrair_codigos


def test_rejeita_notebook_ausente(tmp_path):
    pacote = tmp_path / "sem_notebook.zip"
    with zipfile.ZipFile(pacote, "w") as arquivo:
        arquivo.writestr("arquivo_sem_conteudo.txt", "")
    with pytest.raises(ValueError, match="não encontrado"):
        extrair_codigos(pacote)


def test_rejeita_notebook_grande(tmp_path):
    pacote = tmp_path / "grande.zip"
    with zipfile.ZipFile(pacote, "w") as arquivo:
        arquivo.writestr(CAMINHO, b"x" * 2_000_001)
    with pytest.raises(ValueError, match="tamanho máximo"):
        extrair_codigos(pacote)


def test_rejeita_notebook_com_hash_nao_autorizado(tmp_path):
    pacote = tmp_path / "adulterado.zip"
    with zipfile.ZipFile(pacote, "w") as arquivo:
        arquivo.writestr(CAMINHO, b'{"nbformat":4,"cells":[]}')
    with pytest.raises(ValueError, match="SHA-256"):
        extrair_codigos(pacote)
