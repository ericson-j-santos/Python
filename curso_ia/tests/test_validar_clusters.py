"""Testes do validador offline, sem dependências de nuvem."""
import hashlib
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from validar_clusters import CAMINHO, SHA256_ESPERADO, executar_offline, extrair_codigos


@pytest.fixture(scope="module")
def pacote() -> Path:
    caminho = Path(__file__).parents[3] / "Curso_Inteligencia_Artificial_Estudo_Chaves_Mascaradas.zip"
    if not caminho.is_file():
        pytest.fail("ZIP local necessário para testes E2E; não baixar automaticamente")
    return caminho


def test_positivo_exercicio_real_offline(pacote):
    a = executar_offline(pacote)
    assert a["status"] == "aprovado"
    assert a["exercicio_sha256"] == SHA256_ESPERADO
    assert a["graficos_png_validos"] == 4
    assert a["instancias"] == 150


def test_repeticao_sem_mudar_zip(pacote):
    antes = hashlib.sha256(pacote.read_bytes()).hexdigest()
    a = executar_offline(pacote)
    b = executar_offline(pacote)
    assert a == b
    assert hashlib.sha256(pacote.read_bytes()).hexdigest() == antes


def test_rejeita_codigo_alterado(pacote, tmp_path):
    destino = tmp_path / "adulterado.zip"
    with zipfile.ZipFile(pacote) as entrada, zipfile.ZipFile(destino, "w") as saida:
        trecho = entrada.read(CAMINHO)
        assert b"KMeans" in trecho
        saida.writestr(CAMINHO, trecho.replace(b"KMeans", b"KMeaNS", 1))
    with pytest.raises(ValueError, match="SHA-256"):
        extrair_codigos(destino)


def test_rejeita_arquivo_ausente(tmp_path):
    destino = tmp_path / "vazio.zip"
    with zipfile.ZipFile(destino, "w") as arquivo:
        arquivo.writestr("outro.txt", "sem notebook")
    with pytest.raises(ValueError, match="não encontrado"):
        extrair_codigos(destino)
