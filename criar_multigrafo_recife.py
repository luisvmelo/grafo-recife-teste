#!/usr/bin/env python3
"""
Script para criar multigrafo não direcionado fiel aos dados do Excel.
Cada linha do Excel vira UMA aresta distinta (sem deduplicação).
"""

import argparse
import os
import pandas as pd
import networkx as nx
from pathlib import Path

try:
    from pyvis.network import Network
except Exception:
    Network = None

import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading


def validar_linha(row, idx):
    """Valida se a linha tem vértices válidos. Lança erro se vazio."""
    origem = row['bairro_origem']
    destino = row['bairro_destino']

    if pd.isna(origem) or str(origem).strip() == '':
        raise ValueError(f"Linha {idx}: bairro_origem está vazio ou NaN")
    if pd.isna(destino) or str(destino).strip() == '':
        raise ValueError(f"Linha {idx}: bairro_destino está vazio ou NaN")

    return True


def criar_multigrafo(excel_path):
    """
    Lê Excel e cria MultiGraph não direcionado.
    Retorna: (grafo, dataframe_edges)
    """
    print(f"Lendo arquivo Excel: {excel_path}")

    # Ler Excel
    df = pd.read_excel(excel_path)

    print(f"Total de linhas no Excel: {len(df)}")

    # Verificar colunas esperadas
    colunas_esperadas = ['bairro_origem', 'bairro_destino', 'nome_logradouro', 'distancia_metros']
    for col in colunas_esperadas:
        if col not in df.columns:
            raise ValueError(f"Coluna esperada '{col}' não encontrada no Excel. Colunas disponíveis: {df.columns.tolist()}")

    # Criar MultiGraph não direcionado
    G = nx.MultiGraph()

    edges_list = []

    # Processar cada linha como uma aresta única
    for idx, row in df.iterrows():
        # Validar linha (lança erro se vértice vazio)
        validar_linha(row, idx)

        # Extrair dados (strip apenas espaços em excesso)
        origem = str(row['bairro_origem']).strip()
        destino = str(row['bairro_destino']).strip()

        # Nome do logradouro (pode ser NaN, converter para string vazia)
        nome_logradouro = row['nome_logradouro']
        if pd.isna(nome_logradouro):
            nome_logradouro = ''
        else:
            nome_logradouro = str(nome_logradouro).strip()

        # Peso/distância (manter como float, pode ser NaN)
        peso = row['distancia_metros']
        if not pd.isna(peso):
            try:
                peso = float(peso)
            except (ValueError, TypeError):
                # Se não conseguir converter, manter NaN
                peso = float('nan')
        else:
            peso = float('nan')

        # Edge ID único (índice da linha)
        edge_id = idx

        # Monta rótulo da aresta: "Nome (123 m)" / "Nome" / "123 m"
        label = ""
        if nome_logradouro:
            label = str(nome_logradouro)
        if not pd.isna(peso):
            try:
                peso_int = int(float(peso))
                label = f"{label} ({peso_int} m)" if label else f"{peso_int} m"
            except (ValueError, TypeError):
                pass

        # Adiciona aresta com atributos e 'weight' (fallback 1.0 se NaN)
        G.add_edge(
            origem,
            destino,
            edge_id=edge_id,
            nome=nome_logradouro,
            peso=peso,
            weight=(float(peso) if not pd.isna(peso) else 1.0),
            label=label
        )

        # Guardar para CSV de arestas (agora com 'weight' e 'label')
        edges_list.append({
            'edge_id': edge_id,
            'source': origem,
            'target': destino,
            'nome_logradouro': nome_logradouro,
            'peso': (float(peso) if not pd.isna(peso) else float('nan')),
            'weight': (float(peso) if not pd.isna(peso) else 1.0),
            'label': label
        })

    # Criar DataFrame de arestas
    df_edges = pd.DataFrame(edges_list)

    # Calcula grau (em MultiGraph conta multiarestas)
    degree_dict = dict(G.degree())

    # Define atributos por nó: label = nome do bairro, degree = grau
    for n in G.nodes():
        G.nodes[n]['label'] = n
        G.nodes[n]['degree'] = degree_dict.get(n, 0)

    return G, df_edges


def exportar_arquivos(G, df_edges, output_dir='./out'):
    """Exporta arquivos: edges.csv, nodes.csv, grafo_recife.gexf"""

    # Criar diretório de saída se não existir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1) Exportar edges.csv
    edges_path = os.path.join(output_dir, 'edges.csv')
    df_edges.to_csv(edges_path, index=False, encoding='utf-8')
    print(f"✓ Exportado: {edges_path}")

    # 2) Exportar nodes.csv (com grau)
    nodes_list = []
    for n, data in G.nodes(data=True):
        nodes_list.append({
            'node': n,
            'degree': data.get('degree', 0)
        })
    df_nodes = pd.DataFrame(nodes_list)
    nodes_path = os.path.join(output_dir, 'nodes.csv')
    df_nodes.to_csv(nodes_path, index=False, encoding='utf-8')
    print(f"✓ Exportado: {nodes_path}")

    # 3) Exportar grafo_recife.gexf (formato que preserva multiarestas)
    gexf_path = os.path.join(output_dir, 'grafo_recife.gexf')
    nx.write_gexf(G, gexf_path)
    print(f"✓ Exportado: {gexf_path}")


def _grau_por_pandas(df_edges):
    # grau não-dirigido em multigrafo: conta aparições em source e target
    s = df_edges['source'].value_counts()
    t = df_edges['target'].value_counts()
    grau = (s.add(t, fill_value=0)).astype(int)
    return grau.to_dict()

def visualizar_pyvis(df_edges, output_dir='./out', html_name='grafo_interativo.html', serve=False, port=8000):
    """
    Gera HTML interativo via PyVis a partir do df_edges (sem usar networkx para algoritmos).
    """
    if Network is None:
        print("⚠ Visualização indisponível: instale o PyVis -> pip install pyvis")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    html_path = os.path.join(output_dir, html_name)

    # calcular grau via pandas (conforme regras do projeto)
    graus = _grau_por_pandas(df_edges)

    # nós = união de sources e targets
    nos = sorted(set(df_edges['source']).union(set(df_edges['target'])))

    net = Network(height="850px", width="100%", directed=False, bgcolor="#ffffff")
    net.barnes_hut()  # física para espalhar

    # adicionar nós com label e tamanho por grau
    for n in nos:
        g = int(graus.get(n, 0))
        size = 8 + 3 * (g ** 0.8)
        net.add_node(n, label=str(n), title=f"{n}\nGrau: {g}", value=g, size=size)

    # adicionar arestas com label/weight se existirem no df_edges
    for _, e in df_edges.iterrows():
        u, v = e['source'], e['target']
        title = e.get('label', '')
        weight = e.get('weight', 1.0)
        net.add_edge(u, v, title=title if isinstance(title, str) else '',
                     label=title if isinstance(title, str) and title else None,
                     value=(float(weight) if pd.notna(weight) else 1.0))

    net.write_html(html_path, open_browser=False, notebook=False)
    print(f"✓ Visualização HTML: {html_path}")

    if serve:
        class QuietHandler(SimpleHTTPRequestHandler):
            def log_message(self, *args, **kwargs):
                pass
        cwd = os.getcwd()
        os.chdir(output_dir)
        httpd = HTTPServer(("127.0.0.1", port), QuietHandler)
        url = f"http://127.0.0.1:{port}/{html_name}"
        print(f"▶ Servindo em {url}")
        webbrowser.open(url)
        def _serve():
            try:
                httpd.serve_forever()
            finally:
                os.chdir(cwd)
        threading.Thread(target=_serve, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(
        description='Cria multigrafo não direcionado fiel aos dados do Excel (sem deduplicação)'
    )
    parser.add_argument(
        '--excel_path',
        type=str,
        default='/mnt/c/Users/luise/Downloads/Todas as vias FINAL (1).xlsx',
        help='Caminho para o arquivo Excel (padrão: caminho WSL do arquivo do usuário)'
    )
    parser.add_argument('--viz', action='store_true', help='Gera visualização HTML interativa (PyVis) em out/')
    parser.add_argument('--serve', action='store_true', help='Sobe um servidor local para visualizar o HTML')
    parser.add_argument('--port', type=int, default=8000, help='Porta do servidor local (default: 8000)')

    args = parser.parse_args()

    # Verificar se arquivo existe
    if not os.path.exists(args.excel_path):
        print(f"ERRO: Arquivo não encontrado: {args.excel_path}")
        return

    print("="*60)
    print("CRIANDO MULTIGRAFO NÃO DIRECIONADO - SEM DEDUPLICAÇÃO")
    print("="*60)
    print()

    # Criar multigrafo
    G, df_edges = criar_multigrafo(args.excel_path)

    # Exportar arquivos
    exportar_arquivos(G, df_edges)

    # Visualização interativa (opcional; usa df_edges e pandas, sem algoritmos prontos)
    if args.viz:
        visualizar_pyvis(df_edges, output_dir='./out', html_name='grafo_interativo.html', serve=args.serve, port=args.port)

    print()
    print("="*60)
    print("ESTATÍSTICAS DO GRAFO")
    print("="*60)
    print(f"Número de nós (vértices): {G.number_of_nodes()}")
    print(f"Número de arestas: {G.number_of_edges()}")
    print()

    # Verificar se bate com expectativa
    if G.number_of_edges() == 904:
        print("✓ Número de arestas bate com o esperado (904)")
    else:
        print(f"⚠ Número de arestas ({G.number_of_edges()}) difere do esperado (904)")

    print()
    print("Concluído com sucesso!")


if __name__ == '__main__':
    main()
