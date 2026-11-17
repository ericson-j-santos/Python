"""Gera índice HTML offline, pesquisável e sem links privados a partir de ZIP local."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import zipfile
from urllib.parse import quote

PAGINA = '''<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Índice — Curso de IA</title><style>
:root{font-family:system-ui;color-scheme:light dark}body{max-width:900px;margin:auto;padding:20px}input,select{font:inherit;padding:9px;margin:6px 0;border:1px solid #888;border-radius:7px;background:transparent;color:inherit}input{width:min(95%,380px)}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #8885;padding:9px;text-align:left;overflow-wrap:anywhere}a{color:inherit}small{color:gray}
</style></head><body><h1>Curso de Inteligência Artificial</h1>
<p>Índice local: extraia o ZIP e coloque este HTML na mesma pasta das pastas numeradas.</p>
<label for="busca">Buscar: </label><input id="busca" type="search" placeholder="Nome ou assunto">
<label for="modulo">Módulo: </label><select id="modulo"><option value="">Todos</option></select>
<p id="contagem" role="status"></p><table><thead><tr><th>Arquivo</th><th>Módulo</th><th>Tipo</th></tr></thead><tbody id="itens"></tbody></table>
<script>const dados=__DADOS__;
const busca=document.getElementById("busca"),mod=document.getElementById("modulo"),lista=document.getElementById("itens");
const normalizar=s=>s.normalize("NFD").replace(/[\\u0300-\\u036f]/g,"").toLowerCase();
for(const pasta of [...new Set(dados.map(x=>x.pasta))].sort((a,b)=>parseInt(a)-parseInt(b))){const o=document.createElement("option");o.value=pasta;o.textContent=pasta;mod.append(o)}
function render(){const t=normalizar(busca.value), linhas=dados.filter(x=>(!mod.value||x.pasta===mod.value)&&(!t||normalizar(x.nome+" "+x.pasta).includes(t)));
lista.replaceChildren();const frag=document.createDocumentFragment();for(const x of linhas){const tr=document.createElement("tr"),a=document.createElement("td"),b=document.createElement("td"),c=document.createElement("td"),link=document.createElement("a");link.textContent=x.nome;link.href=x.href;a.append(link);b.textContent=x.pasta;c.textContent=x.tipo;tr.append(a,b,c);frag.append(tr)}lista.append(frag);document.getElementById("contagem").textContent=linhas.length+" de "+dados.length+" arquivos"}
busca.addEventListener("input",render);mod.addEventListener("change",render);render();
</script></body></html>'''


def gerar(pacote: Path, saida: Path) -> dict:
    if saida.exists():
        raise FileExistsError("Índice já existe; não sobrescrever")
    with zipfile.ZipFile(pacote) as arquivo:
        nomes = [x.filename for x in arquivo.infolist() if not x.is_dir() and "/" in x.filename and not x.filename.endswith("Zone.Identifier")]
    dados = []
    for nome in nomes:
        pasta, arquivo = nome.split("/", 1)
        if not pasta[:1].isdigit():
            continue
        dados.append({"pasta": pasta, "nome": arquivo, "tipo": Path(arquivo).suffix.lstrip(".").upper(),
                      "href": "/".join(quote(p) for p in nome.split("/"))})
    dados.sort(key=lambda x: (int(x["pasta"].split(".", 1)[0]), x["nome"].casefold()))
    serializado = json.dumps(dados, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    saida.write_text(PAGINA.replace("__DADOS__", serializado), encoding="utf-8")
    return {"arquivos": len(dados), "modulos": len(set(x["pasta"] for x in dados)), "indice": str(saida)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", required=True, type=Path)
    p.add_argument("--saida", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(gerar(a.zip, a.saida), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
