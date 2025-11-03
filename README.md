# Grafo Recife - Teste

Multigrafo não direcionado das vias de Recife, preservando todas as arestas paralelas e auto-laços.

## Descrição

Este projeto converte dados de vias entre bairros de Recife (arquivo Excel) em um multigrafo não direcionado usando NetworkX. Cada linha do Excel vira uma aresta distinta, sem deduplicação ou agregação.

## Características do Grafo

- **Tipo:** MultiGraph não direcionado (suporta arestas paralelas)
- **Nós:** 88 bairros de Recife
- **Arestas:** 904 vias (cada linha do Excel = 1 aresta)
- **Atributos de nós:**
  - `label`: nome do bairro
  - `degree`: grau do vértice (contando multiarestas)
- **Atributos de arestas:**
  - `edge_id`: identificador único da aresta
  - `nome`: nome do logradouro
  - `peso`: distância em metros (float)
  - `weight`: peso para visualização no Gephi (distância ou 1.0 se NaN)
  - `label`: rótulo formatado "Nome (123 m)"

## Arquivos Gerados

### Saídas (pasta `out/`)
- **edges.csv**: lista de todas as arestas com atributos
- **nodes.csv**: lista de nós com grau
- **grafo_recife.gexf**: formato GEXF para importar no Gephi
- **grafo_interativo.html**: visualização HTML interativa (gerado com `--viz`)

## Uso

### Requisitos
- Python 3.10+
- pandas
- openpyxl
- networkx
- pyvis (para visualização HTML interativa)
- streamlit (para app interativo com algoritmos)

### Instalação

```bash
# Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

### Executar

**1. Gerar dados do grafo (criar_multigrafo_recife.py):**

```bash
# Básico: apenas gerar CSVs e GEXF
python criar_multigrafo_recife.py

# Com caminho customizado
python criar_multigrafo_recife.py --excel_path /caminho/para/arquivo.xlsx

# Gerar visualização HTML interativa
python criar_multigrafo_recife.py --viz

# Gerar HTML e abrir em servidor local (http://127.0.0.1:8000)
python criar_multigrafo_recife.py --viz --serve

# Customizar porta do servidor
python criar_multigrafo_recife.py --viz --serve --port 8080
```

**2. App Streamlit interativo (app.py):**

```bash
streamlit run app.py
```

O app oferece:
- Visualização interativa do grafo
- Dijkstra: caminho mais curto entre dois bairros
- Kruskal: árvore geradora mínima (AGM)
- Algoritmos implementados conforme slides das aulas (sem usar funções prontas)

## Visualização

### HTML Interativo (PyVis)

A visualização HTML interativa oferece:
- Grafo interativo com física de força (arraste nós, zoom, pan)
- Tamanho dos nós proporcional ao grau (calculado via pandas)
- Hover sobre nós mostra: nome do bairro + grau
- Hover sobre arestas mostra: nome da via + distância
- Renderizado com vis.js

Para gerar e visualizar:
```bash
python criar_multigrafo_recife.py --viz --serve
```

Depois abra: http://127.0.0.1:8000/grafo_interativo.html

### Gephi

1. Abrir Gephi
2. Arquivo → Abrir → Selecionar `out/grafo_recife.gexf`
3. Configurar visualização:
   - Tamanho dos nós: usar atributo `degree`
   - Espessura das arestas: usar atributo `weight`
   - Labels: ativar para ver nomes de bairros e vias

## Estrutura do Projeto

```
grafo-recife-teste/
├── criar_multigrafo_recife.py   # Script principal
├── out/                          # Arquivos gerados
│   ├── edges.csv
│   ├── nodes.csv
│   ├── grafo_recife.gexf
│   └── grafo_interativo.html    # Visualização HTML (com --viz)
├── lib/                          # Bibliotecas JS para visualização
│   ├── vis-9.1.2/
│   ├── tom-select/
│   └── bindings/
├── requirements.txt
├── .gitignore
└── README.md
```

## Dados de Entrada

Esperado um arquivo Excel com as seguintes colunas:
- `bairro_origem`: vértice de origem
- `bairro_destino`: vértice de destino
- `nome_logradouro`: nome da via
- `distancia_metros`: distância em metros

## Validações

- Não remove auto-laços (origem = destino)
- Não normaliza nomes de bairros
- Não colapsa arestas A-B e B-A
- Não soma pesos de arestas paralelas
- Preserva NaN em pesos (com fallback weight=1.0)
- Lança erro se vértices vazios

## Algoritmos Implementados

### Dijkstra (Caminho Mais Curto)
Implementação conforme **AULA 05 - CAMINHOS MAIS CURTO_DIJKSTRA**:
- Vetores: `d[v]` (distância estimada), `p[v]` (predecessor), `aberto[v]` (boolean)
- Inicialização: `d[s]=0`, `d[v]=∞` para v≠s
- Loop principal com relaxamento de arestas
- Fila de prioridade (heap) para eficiência
- Reconstrução do caminho via predecessores

### Kruskal (Árvore Geradora Mínima)
Implementação conforme **AULA 11 e 12 - AMG/KRUSKAL**:
- Union-Find com path compression e union by rank
- Ordena arestas por peso crescente
- Processa arestas evitando ciclos
- AGM final tem |V|-1 arestas
- Algoritmo guloso formando floresta → árvore

## Notas sobre Implementação

- **Cálculo de graus:** usa pandas (não networkx) conforme regras do projeto
- **Visualização:** usa PyVis (não usa algoritmos prontos de libs externas)
- **Multigrafo:** preserva todas as arestas paralelas e auto-laços
- **Algoritmos:** Dijkstra e Kruskal implementados do zero (sem `nx.shortest_path` ou `nx.minimum_spanning_tree`)

## Autor

Criado com Claude Code
