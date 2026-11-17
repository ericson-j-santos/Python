# Curso de IA — Validação offline

Código próprio para validar **sem rede e sem serviços pagos** o exercício `Clusters.ipynb` do módulo 3.

## Origem e limites

O exercício vem de um curso privado do usuário; **seus arquivos, vídeos, materiais, notebooks originais, credenciais e dados do Drive não são publicados neste repositório**.
Este código somente acessa um ZIP local informado explicitamente. Foi gerado para esta validação assistida, sem downloads automáticos e sem alterações no curso.

- Requisitos já instalados: Python, numpy, scipy, scikit-learn, matplotlib e pytest (para testes).
- O código verifica o SHA-256 do `Clusters.ipynb` antes de executá-lo.
- Executa todas as células de código do exercício original em processo Python separado, com `matplotlib` sem janela, `socket` bloqueado, ambiente sanitizado e tempo máximo de 35 segundos.
- Registra métricas dos três agrupadores (K-means, DBSCAN e hierárquico) e comprova a geração de quatro gráficos PNG.
- Os testes verificam cenário positivo, adulteração do notebook, ausência de notebook e repetição sem alterar o ZIP.
- O bloqueio de rede é aplicado no Python; não substitui isolamento por firewall/sistema operacional para execução de **outros** códigos não auditados. O script só autoriza o digest fixado deste exercício.

## Executar

Coloque o ZIP do curso **fora do Git**. Exemplo, depois de preparar Python e as bibliotecas:

```bash
python curso_ia/validar_clusters.py --zip /caminho/local/Curso_Inteligencia_Artificial_Estudo_Chaves_Mascaradas.zip --saida evidencia.json
```

Para testes automatizados, mantenha o ZIP local na posição definida no `tests/test_validar_clusters.py` ou adapte o caminho sem publicar o material. **Não configure CI pública para executar com o ZIP privado.**

## Critério de aceite

O relatório deve apontar 150 amostras, quatro atributos, três grupos K-means, três grupos hierárquicos, quatro gráficos PNG válidos, SHA-256 conferido e zero alteração no ZIP. A validação do código em repositório não constitui execução nos computadores do usuário.

## Ferramentas próprias de organização

`mascarar_chaves.py`: gera **nova** cópia ZIP sem sobrescrever a origem e mascara padrões específicos de token nos notebooks. Não substitui auditoria completa de segredos.

`gerar_indice_ia.py`: cria índice HTML com busca local, sem conteúdos dos documentos ou links do Drive.

```bash
python curso_ia/mascarar_chaves.py --origem curso_original.zip --destino curso_mascarado.zip
python curso_ia/gerar_indice_ia.py --zip curso_mascarado.zip --saida indice.html
```

As ferramentas usam apenas a biblioteca padrão do Python. Não armazenar ou publicar os ZIPs no Git.
