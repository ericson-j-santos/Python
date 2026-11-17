"""Testes públicos: criam dados artificiais, sem material do curso."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from projeto_final import renda


def criar_dados_sinteticos(destino: Path, tamanho: int = 140) -> Path:
    destino.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(451)
    total = tamanho * 3
    idade = rng.integers(18, 72, total)
    estudos = rng.integers(4, 17, total)
    horas = rng.integers(20, 65, total)
    escore = idade * .025 + estudos * .35 + horas * .035 + rng.normal(0, 1.4, total)
    renda_alta = escore > np.median(escore)
    df = pd.DataFrame({
        "age": idade, "workclass": rng.choice(["Private", "?", "Local-gov"], total),
        "fnlwgt": rng.integers(80_000, 300_000, total),
        "education": np.where(estudos >= 12, "Bachelors", "HS-grad"),
        "educational-num": estudos,
        "marital-status": rng.choice(["Married-civ-spouse", "Never-married"], total),
        "occupation": rng.choice(["Craft-repair", "Sales", "?"], total),
        "relationship": rng.choice(["Husband", "Not-in-family", "Unmarried"], total),
        "race": rng.choice(["White", "Black"], total),
        "gender": rng.choice(["Male", "Female"], total),
        "capital-gain": rng.choice([0, 0, 1, 2000], total),
        "capital-loss": rng.choice([0, 0, 30, 500], total),
        "hours-per-week": horas,
        "native-country": rng.choice(["United-States", "Canada", "?"], total),
        "income": np.where(renda_alta, ">50K", "<=50K"),
    }, columns=list(renda.COLUNAS))
    # Reequilibra as classes de cada partição sem consultar o conjunto de teste na seleção.
    for i, nome in enumerate(("train.csv", "validation.csv", "test.csv")):
        bloco = df.iloc[i * tamanho:(i + 1) * tamanho].copy()
        if bloco["income"].nunique() != 2:
            raise AssertionError("Amostra sintética inesperadamente sem as duas classes")
        bloco.to_csv(destino / nome, index=False)
    return destino


def test_campos_e_labels_invalidos_bloqueados(tmp_path):
    p = criar_dados_sinteticos(tmp_path)
    df = pd.read_csv(p / "train.csv")
    df.drop(columns=["income"]).to_csv(p / "sem_alvo.csv", index=False)
    with pytest.raises(ValueError, match="Esquema inválido"):
        renda.ler_csv(p / "sem_alvo.csv")
    df.loc[0, "income"] = "INDEFINIDO"
    df.to_csv(p / "alvo_invalido.csv", index=False)
    with pytest.raises(ValueError, match="Classes inválidas"):
        renda.ler_csv(p / "alvo_invalido.csv")


def test_nao_publicar_raca_genero_nem_alvo_como_entrada(tmp_path):
    p = criar_dados_sinteticos(tmp_path)
    x = renda.preparar_x(renda.ler_csv(p / "train.csv"))
    assert not {"race", "gender", "income", "education"} & set(x.columns)
    assert x["workclass"].isna().any()
    assert x["occupation"].isna().any()
    assert x["capital-gain"].min() >= 0


def test_rejeitar_numerico_negativo(tmp_path):
    p = criar_dados_sinteticos(tmp_path)
    df = renda.ler_csv(p / "train.csv")
    df.loc[0, "capital-gain"] = -1
    with pytest.raises(ValueError, match="negativo"):
        renda.preparar_x(df)


def test_limite_nao_muda_com_repeticao():
    y = np.asarray([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
    p = np.asarray([.2, .4, .6, .75, .43, .28, .85, .34, .91, .12])
    a = renda.escolher_limiar(y, p)
    b = renda.escolher_limiar(y, p)
    assert a == b and 0 <= a["f1"] <= 1
    assert a["matriz_confusao"] and len(a["matriz_confusao"]) == 2


def test_teste_lido_so_depois_da_escolha_e_arquivos_inalterados(tmp_path, monkeypatch):
    p = criar_dados_sinteticos(tmp_path / "dados")
    antes = {nome: hashlib.sha256((p / nome).read_bytes()).hexdigest()
             for nome in ("train.csv", "validation.csv", "test.csv")}
    eventos = []
    ler_original = renda.ler_csv
    escolher_original = renda.escolher_limiar
    def ler_observavel(caminho):
        eventos.append("ler:" + caminho.name)
        return ler_original(caminho)
    def escolher_observavel(*args, **kwargs):
        eventos.append("escolher")
        return escolher_original(*args, **kwargs)
    def so_logistica():
        return {"regressao_logistica": Pipeline([
            ("preparo", renda.processador(True)),
            ("classificador", LogisticRegression(max_iter=300, random_state=renda.SEMENTE)),
        ])}
    monkeypatch.setattr(renda, "ler_csv", ler_observavel)
    monkeypatch.setattr(renda, "escolher_limiar", escolher_observavel)
    monkeypatch.setattr(renda, "modelos", so_logistica)
    monkeypatch.setattr(renda, "importancia_permutacao", lambda *args: [])
    a = renda.realizar(p)
    assert eventos.index("escolher") < eventos.index("ler:test.csv")
    assert eventos.count("ler:test.csv") == 1
    assert a["divisao"]["teste"] == 140
    assert a["fonte_sha256"] == antes
    destino = tmp_path / "saida"
    arquivos = renda.escrever_resultados(a, destino)
    assert len(arquivos) == 2 and all(x.exists() for x in arquivos)
    with pytest.raises(FileExistsError):
        renda.escrever_resultados(a, destino)
    assert {nome: hashlib.sha256((p / nome).read_bytes()).hexdigest() for nome in antes} == antes


def test_dados_ausentes_falham_sem_mudar_arquivo(tmp_path):
    p = criar_dados_sinteticos(tmp_path)
    (p / "test.csv").unlink()
    with pytest.raises(FileNotFoundError, match="test.csv"):
        renda.realizar(p)
