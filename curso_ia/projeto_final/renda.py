"""Desafio final Adult Income: avaliação reprodutível, local e sem publicar dados.

O conjunto test.csv é lido exclusivamente depois de escolhidos modelo e limiar
com base em train.csv e validation.csv. Nenhum arquivo de entrada é alterado.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

ALVO = "income"
POSITIVO = ">50K"
NEGATIVO = "<=50K"
COLUNAS = (
    "age", "workclass", "fnlwgt", "education", "educational-num", "marital-status",
    "occupation", "relationship", "race", "gender", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
)
NUMERICAS = ("age", "fnlwgt", "educational-num", "capital-gain", "capital-loss", "hours-per-week")
CATEGORICAS = ("workclass", "marital-status", "occupation", "relationship", "native-country")
VARIAVEIS = NUMERICAS + CATEGORICAS
NAO_USADAS = ("education", "race", "gender")
SEMENTE = 42
LIMIARES = tuple(round(float(x), 3) for x in np.arange(0.25, 0.701, 0.025))


def sha256(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def ler_csv(caminho: Path) -> pd.DataFrame:
    if not caminho.is_file():
        raise FileNotFoundError(f"CSV ausente: {caminho.name}")
    df = pd.read_csv(caminho)
    if tuple(df.columns) != COLUNAS:
        raise ValueError(f"Esquema inválido em {caminho.name}")
    if df.empty or df.duplicated().all():
        raise ValueError(f"Dados vazios ou inválidos: {caminho.name}")
    if df[ALVO].isna().any() or not set(df[ALVO].unique()).issubset({NEGATIVO, POSITIVO}):
        raise ValueError(f"Classes inválidas em {caminho.name}")
    if df[ALVO].nunique() != 2:
        raise ValueError(f"Amostra sem as duas classes em {caminho.name}")
    for coluna in NUMERICAS:
        df[coluna] = pd.to_numeric(df[coluna], errors="raise")
        if not np.isfinite(df[coluna].dropna().to_numpy()).all():
            raise ValueError(f"Valor numérico não finito: {coluna}")
    return df


def preparar_x(df: pd.DataFrame) -> pd.DataFrame:
    """Preserva as colunas, trata '?' e transforma ganhos/perdas de cauda longa."""
    x = df.loc[:, VARIAVEIS].copy()
    for coluna in CATEGORICAS:
        x[coluna] = x[coluna].replace("?", np.nan)
    for coluna in ("capital-gain", "capital-loss", "fnlwgt"):
        if x[coluna].lt(0).any():
            raise ValueError(f"Valor negativo inesperado em {coluna}")
        x[coluna] = np.log1p(x[coluna])
    return x


def preparar_y(df: pd.DataFrame) -> np.ndarray:
    return (df[ALVO] == POSITIVO).astype(int).to_numpy()


def processador(linear: bool) -> ColumnTransformer:
    num = Pipeline([
        ("ausentes", SimpleImputer(strategy="median")),
        ("escala", RobustScaler()) if linear else ("escala", "passthrough"),
    ])
    cat = Pipeline([
        ("ausentes", SimpleImputer(strategy="most_frequent")),
        ("categorias", OneHotEncoder(handle_unknown="ignore") if linear else
         OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer([
        ("numericas", num, list(NUMERICAS)),
        ("categoricas", cat, list(CATEGORICAS)),
    ])


def modelos() -> dict[str, Pipeline]:
    return {
        "regressao_logistica": Pipeline([
            ("preparo", processador(True)),
            ("classificador", LogisticRegression(max_iter=600, random_state=SEMENTE)),
        ]),
        "floresta_aleatoria": Pipeline([
            ("preparo", processador(True)),
            ("classificador", RandomForestClassifier(
                n_estimators=140, min_samples_leaf=3, n_jobs=2, random_state=SEMENTE)),
        ]),
        "boosting_histogramas": Pipeline([
            ("preparo", processador(False)),
            ("classificador", HistGradientBoostingClassifier(
                categorical_features=[False] * len(NUMERICAS) + [True] * len(CATEGORICAS),
                max_iter=190, max_leaf_nodes=23, learning_rate=0.07,
                l2_regularization=1, random_state=SEMENTE)),
        ]),
    }


def metricas(y: np.ndarray, p: np.ndarray, limiar: float) -> dict[str, Any]:
    classe = (p >= limiar).astype(int)
    cm = confusion_matrix(y, classe, labels=[0, 1])
    return {
        "f1": round(float(f1_score(y, classe, zero_division=0)), 6),
        "precisao": round(float(precision_score(y, classe, zero_division=0)), 6),
        "revocacao": round(float(recall_score(y, classe, zero_division=0)), 6),
        "acuracia": round(float(accuracy_score(y, classe)), 6),
        "acuracia_balanceada": round(float(balanced_accuracy_score(y, classe)), 6),
        "roc_auc": round(float(roc_auc_score(y, p)), 6),
        "matriz_confusao": cm.tolist(),
        "limiar": float(limiar),
    }


def escolher_limiar(y: np.ndarray, probabilidades: np.ndarray) -> dict[str, Any]:
    """Limiar otimizado exclusivamente nos dados de validação."""
    opcoes = [metricas(y, probabilidades, t) for t in LIMIARES]
    # Desempate explícito: maior F1, maior precisão e limiar mais próximo de 0.5.
    return max(opcoes, key=lambda m: (m["f1"], m["precisao"], -abs(m["limiar"] - 0.5)))


def importancia_permutacao(
    modelo: Pipeline, x: pd.DataFrame, y: np.ndarray, limiar: float,
) -> list[dict[str, Any]]:
    """Explicação global: perda de F1 quando uma entrada é embaralhada."""
    rng = np.random.default_rng(SEMENTE)
    pontuacao_referencia = f1_score(y, modelo.predict_proba(x)[:, 1] >= limiar)
    importancia = []
    for coluna in VARIAVEIS:
        diferencas = []
        for _ in range(2):
            x_alt = x.copy()
            x_alt[coluna] = x_alt[coluna].iloc[rng.permutation(len(x))].to_numpy()
            nova = f1_score(y, modelo.predict_proba(x_alt)[:, 1] >= limiar)
            diferencas.append(float(pontuacao_referencia - nova))
        importancia.append({"atributo": coluna, "perda_f1": round(float(np.mean(diferencas)), 6)})
    return sorted(importancia, key=lambda z: z["perda_f1"], reverse=True)


def realizar(diretorio: Path) -> dict[str, Any]:
    """Treina/seleciona em train+validation e só depois lê o test para avaliar."""
    caminho_treino = diretorio / "train.csv"
    caminho_val = diretorio / "validation.csv"
    caminho_teste = diretorio / "test.csv"
    origem = {nome: sha256(diretorio / nome) for nome in ("train.csv", "validation.csv")}
    if not caminho_teste.is_file():
        raise FileNotFoundError("CSV ausente: test.csv")
    treino = ler_csv(caminho_treino)
    validacao = ler_csv(caminho_val)
    x_train, y_train = preparar_x(treino), preparar_y(treino)
    x_val, y_val = preparar_x(validacao), preparar_y(validacao)
    if not set(y_train) == {0, 1} or not set(y_val) == {0, 1}:
        raise ValueError("Treino ou validação sem ambas as classes")

    tentativas = []
    referencia = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    met_ref = metricas(y_val, referencia.predict_proba(x_val)[:, 1], 0.5)
    tentativas.append({"modelo": "maioria", "etapa": "referencia", **met_ref})
    candidatos = modelos()
    vencedor = None
    selecao = None
    for nome, modelo in candidatos.items():
        modelo.fit(x_train, y_train)
        probabilidades = modelo.predict_proba(x_val)[:, 1]
        met_padrao = metricas(y_val, probabilidades, 0.5)
        met_otim = escolher_limiar(y_val, probabilidades)
        tentativas.append({"modelo": nome, "etapa": "limiar_padrao", **met_padrao})
        tentativas.append({"modelo": nome, "etapa": "ajuste_limiar_validacao", **met_otim})
        if selecao is None or met_otim["f1"] > selecao["f1"]:
            selecao = {"modelo": nome, **met_otim}
            vencedor = modelo
    assert vencedor is not None and selecao is not None
    explicacao = importancia_permutacao(vencedor, x_val, y_val, selecao["limiar"])

    # Primeiro acesso ao conteúdo de test.csv: após escolher modelo e limiar.
    origem["test.csv"] = sha256(caminho_teste)
    teste = ler_csv(caminho_teste)
    x_test, y_test = preparar_x(teste), preparar_y(teste)
    final = metricas(y_test, vencedor.predict_proba(x_test)[:, 1], selecao["limiar"])
    depois = {nome: sha256(diretorio / nome) for nome in origem}
    if origem != depois:
        raise RuntimeError("Dados de origem alterados durante a execução")
    return {
        "situacao": "validado_localmente", "metrica_principal": "f1_classe_renda_maior_50k",
        "divisao": {"treinamento": len(treino), "validacao": len(validacao), "teste": len(teste)},
        "fonte_sha256": origem, "tratamento_ausentes": "'?' convertido para NaN; imputação aprende apenas no treinamento",
        "outliers": "log1p em ganhos/perdas e fnlwgt; escala robusta para regressão logística",
        "selecao_atributos": {"utilizados": list(VARIAVEIS), "excluidos": list(NAO_USADAS),
                               "criterio": "education redundante com educational-num; race/gender reservados por risco de vies"},
        "tabela_ajustes_validacao": tentativas,
        "vencedor_validacao": selecao,
        "explicacao_global_validacao": explicacao,
        "avaliacao_final_teste": final,
        "limites": ["Não usar para decisões individuais de renda, contratação ou crédito",
                    "A retirada de raça e gênero não garante ausência de vieses em variáveis correlacionadas",
                    "Possíveis linhas repetidas entre arquivos de origem; scores podem estar levemente otimistas",
                    "A seleção foi feita na validação, sem ajuste pelo conjunto de teste"],
        "bibliotecas": {"sklearn": __import__("sklearn").__version__, "pandas": pd.__version__},
    }


def escrever_resultados(resultado: dict[str, Any], destino: Path) -> list[Path]:
    if destino.exists() and any(destino.iterdir()):
        raise FileExistsError("Destino não está vazio; use diretório novo para impedir sobrescrita")
    destino.mkdir(parents=True, exist_ok=True)
    caminho_json = destino / "resultado.json"
    caminho_md = destino / "relatorio.md"
    caminho_json.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    linhas = [
        "# Projeto final — previsão da faixa de renda (Adult Income)",
        "", "**Objetivo:** maximizar o F1 Score da classe renda >50K usando dados do curso.",
        "", "## Resultado principal", "",
        f"- **Modelo selecionado:** {resultado['vencedor_validacao']['modelo']}",
        f"- **F1 na validação:** {resultado['vencedor_validacao']['f1']:.4f}",
        f"- **Limiar escolhido na validação:** {resultado['vencedor_validacao']['limiar']:.3f}",
        f"- **F1 no teste final:** {resultado['avaliacao_final_teste']['f1']:.4f}",
        f"- **Precisão no teste:** {resultado['avaliacao_final_teste']['precisao']:.4f}",
        f"- **Revocação no teste:** {resultado['avaliacao_final_teste']['revocacao']:.4f}",
        f"- **ROC AUC no teste:** {resultado['avaliacao_final_teste']['roc_auc']:.4f}",
        "", "## Histórico de tentativas — apenas validação", "",
        "| Modelo | Etapa | Limiar | F1 | Precisão | Revocação |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for m in resultado["tabela_ajustes_validacao"]:
        linhas.append(f"| {m['modelo']} | {m['etapa']} | {m['limiar']:.3f} | {m['f1']:.4f} | {m['precisao']:.4f} | {m['revocacao']:.4f} |")
    linhas += ["", "## Matriz de confusão — teste final", "",
               "Linhas: realidade (<=50K, >50K); colunas: previsão (<=50K, >50K).", "",
               f"`{resultado['avaliacao_final_teste']['matriz_confusao']}`", "",
               "## Explicabilidade global — importância por permutação", "",
               "A importância mede quanto o F1 de validação diminui ao embaralhar uma variável; não explica causalidade.", "",
               "| Atributo | Perda média de F1 na validação |", "|---|---:|"]
    for item in resultado["explicacao_global_validacao"]:
        linhas.append(f"| {item['atributo']} | {item['perda_f1']:.4f} |")
    linhas.extend(["", "## Pré-processamento e segurança", "",
                   "- Valores `?` tratados como ausentes, imputação ajustada **só no treinamento**.",
                   "- Codificação de categorias; logaritmo `log1p` nos valores extremos e escala robusta quando cabível.",
                   "- `education` removida por redundância com `educational-num`; `race` e `gender` excluídas das entradas por cautela.",
                   "- Dados brutos não publicados; o código não instala bibliotecas nem acessa API externa.",
                   "- `test.csv` é usado **uma vez, somente após seleção** por validação.",
                   "- Este experimento é acadêmico e **não está aprovado** para decisões sobre pessoas.",
                   "", "## Limitações", ""])
    linhas.extend(f"- {s}" for s in resultado["limites"])
    caminho_md.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return [caminho_json, caminho_md]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dados", required=True, type=Path, help="Diretório dos três CSV originais")
    p.add_argument("--saida", required=True, type=Path, help="Diretório novo para métricas agregadas")
    args = p.parse_args()
    if args.saida.exists() and any(args.saida.iterdir()):
        p.error("Diretório de saída deve estar vazio")
    resultado = realizar(args.dados)
    arquivos = escrever_resultados(resultado, args.saida)
    print(json.dumps({"status": "concluido", "f1_teste": resultado["avaliacao_final_teste"]["f1"],
                      "modelo": resultado["vencedor_validacao"]["modelo"],
                      "arquivos": [str(x) for x in arquivos]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
