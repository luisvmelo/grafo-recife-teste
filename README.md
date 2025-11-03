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

## Uso

### Requisitos
- Python 3.10+
- pandas
- openpyxl
- networkx

### Instalação

```bash
# Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install pandas openpyxl networkx
```

### Executar

```bash
# Com caminho padrão (WSL)
python criar_multigrafo_recife.py

# Com caminho customizado
python criar_multigrafo_recife.py --excel_path /caminho/para/arquivo.xlsx
```

## Visualização no Gephi

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
│   └── grafo_recife.gexf
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

## Autor

Criado com Claude Code
