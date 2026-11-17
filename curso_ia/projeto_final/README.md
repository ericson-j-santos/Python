# Projeto final — classificação de faixa de renda (Adult Income)

Projeto **acadêmico** derivado do enunciado `desafio.pdf` do curso. O objetivo é obter o melhor **F1 Score** possível para a classe `>50K`, sem usar `test.csv` para selecionar o modelo ou ajustar o limiar de decisão.

## Executar

Requer Python 3.12+, `numpy`, `pandas` e `scikit-learn`. Não instala dependências automaticamente, não usa API e não envia dados à rede. Copie somente o **código autoral** deste repositório; mantenha os CSVs privados em sua máquina.

```bash
python -m curso_ia.projeto_final.renda --dados "/diretorio/privado/com/csv" --saida "/diretorio/novo/resultado"
```

O diretório `--dados` deve conter `train.csv`, `validation.csv` e `test.csv`. O diretório `--saida` precisa ser novo ou estar vazio; o script recusa sobrescrever um resultado existente. A saída contém **somente métricas agregadas** nos arquivos `resultado.json` e `relatorio.md`, nunca registros individuais, modelo serializado ou CSVs de entrada.

## O que é executado

1. Confere a estrutura e os nomes de colunas dos arquivos e registra SHA-256 dos três CSVs.
2. Substitui `?` por ausente nas categorias; aplica imputação, transformação logarítmica para valores muito assimétricos e escala robusta quando necessário. As transformações são treinadas apenas no conjunto `train.csv`.
3. Remove do modelo a variável `education` (redundante com `educational-num`) e deixa `race` e `gender` fora da previsão, por cautela. Isso **não prova ausência de discriminação**, pois outras entradas podem funcionar como substitutas.
4. Compara baseline de classe majoritária, regressão logística, floresta aleatória e *histogram gradient boosting* com **os dados de validação**. Ajusta o limiar de classificação para maximizar F1 somente na validação.
5. Executa explicação global por permutação dos atributos no conjunto de validação: a maior queda de F1 ao embaralhar uma entrada sugere maior dependência dela. Isso **não mede causalidade**.
6. **Só depois da escolha**, lê `test.csv` e faz a **avaliação final única**. Calcula F1, precisão, revocação, acurácia, acurácia balanceada, ROC AUC e matriz de confusão.
7. Confere novamente os hashes dos três CSVs para detectar alteração indevida.

## Resultado medido em 16/11/2026 (BRT), ambiente de análise isolado

| Modelo | F1 — limiar 0,50 (validação) | F1 — melhor limiar (validação) | Limiar |
|---|---:|---:|---:|
| Classe majoritária | 0,0000 | — | 0,50 |
| Regressão logística | 0,6442 | 0,6759 | 0,350 |
| Floresta aleatória | 0,6882 | 0,7140 | 0,375 |
| Boosting de histogramas | 0,7148 | **0,7295** | **0,400** |

- **F1 final no teste:** **0,7303**. Precisão **0,7207**, revocação **0,7402**, ROC AUC **0,9333**.
- Matriz de confusão final (ordem linhas/colunas `<=50K`, `>50K`): `[[5093, 498], [451, 1285]]`.
- Na explicação por permutação, os maiores impactos globais na validação foram `capital-gain`, `marital-status`, `educational-num`, `age` e `occupation`.
- Amostras: treinamento **34.189**, validação **7.326**, teste **7.327** (total **48.842**). O slide do curso descreve 45.225 instâncias; a divergência foi registrada sem alterar os arquivos.

**Limites e riscos:** há registros idênticos em diferentes partições de origem, o que pode deixar as métricas um pouco otimistas. A seleção foi feita apenas na validação, e o conjunto de teste não foi reutilizado para melhorar o modelo. O exercício não é apropriado para decisões de contratação, crédito, direitos ou avaliação individual de pessoas. Em nenhum momento foram publicados os CSVs, o ZIP do curso ou credenciais.

## Qualidade e verificação

- Os testes públicos utilizam dados **sintéticos**, criados dentro do teste, e podem ser executados no GitHub Actions sem o conteúdo original do curso.
- A execução completa com os CSVs originais foi realizada **fora do GitHub**, em ambiente local desta análise. CI público não é prova da reprodução dos dados privados.
- Não utilize testes em ambiente de produção ou dados identificáveis sem permissão.
- Os resultados dependem das versões instaladas e do formato exato dos arquivos. Comparações exigem preservar a separação treino/validação/teste.
