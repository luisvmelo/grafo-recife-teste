#!/usr/bin/env python3
"""
App Streamlit minimalista para grafo de vias de Recife.
UI: apenas canvas grande + sidebar com Bairro 1, Bairro 2, Método.
Algoritmos: Dijkstra, BFS/DFS, Ranking por grau (implementações manuais).
"""

import streamlit as st
import pandas as pd
import os
from collections import defaultdict, deque
import heapq

try:
    from pyvis.network import Network
except ImportError:
    Network = None


# ============================================================================
# UNION-FIND (para Kruskal, se necessário no futuro)
# ============================================================================

class UnionFind:
    def __init__(self, nodes):
        self.parent = {n: n for n in nodes}
        self.rank = {n: 0 for n in nodes}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        return True


# ============================================================================
# DIJKSTRA - Caminho mais curto
# Referência: AULA 05 - CAMINHOS MAIS CURTO_DIJKSTRA
# ============================================================================

def dijkstra(adj, s, t, nodes):
    """
    Dijkstra manual: vetores d, p, aberto + fila de prioridade.
    Retorna: (distancia, caminho_nos, caminho_arestas)
    """
    INF = float('inf')
    d = {v: INF for v in nodes}
    p = {v: None for v in nodes}
    p_edge = {v: None for v in nodes}
    aberto = {v: True for v in nodes}

    d[s] = 0
    heap = [(0, s)]

    while heap:
        dist_u, u = heapq.heappop(heap)
        if not aberto[u]:
            continue
        aberto[u] = False
        if u == t:
            break

        if u in adj:
            for v, peso, edge_id, nome in adj[u]:
                if aberto[v]:
                    nova_dist = d[u] + peso
                    if nova_dist < d[v]:
                        d[v] = nova_dist
                        p[v] = u
                        p_edge[v] = (edge_id, nome, peso, u)
                        heapq.heappush(heap, (nova_dist, v))

    if d[t] == INF:
        return (INF, [], [])

    caminho_nos = []
    caminho_arestas = []
    v = t
    while v is not None:
        caminho_nos.append(v)
        if p[v] is not None:
            edge_id, nome, peso, u = p_edge[v]
            caminho_arestas.append((u, v, peso, nome, edge_id))
        v = p[v]

    caminho_nos.reverse()
    caminho_arestas.reverse()
    return (d[t], caminho_nos, caminho_arestas)


# ============================================================================
# BFS - Busca em Largura (Árvore de travessia)
# ============================================================================

def bfs_tree(adj, raiz, nodes):
    """
    BFS manual a partir de raiz.
    Retorna: (ordem_visita, arestas_arvore)
    arestas_arvore = [(u, v, peso, nome, edge_id), ...]
    """
    visitado = {n: False for n in nodes}
    ordem = []
    arestas_arvore = []
    fila = deque([raiz])
    visitado[raiz] = True

    while fila:
        u = fila.popleft()
        ordem.append(u)

        if u in adj:
            for v, peso, edge_id, nome in adj[u]:
                if not visitado[v]:
                    visitado[v] = True
                    fila.append(v)
                    arestas_arvore.append((u, v, peso, nome, edge_id))

    return (ordem, arestas_arvore)


# ============================================================================
# DFS - Busca em Profundidade (Árvore de travessia)
# ============================================================================

def dfs_tree(adj, raiz, nodes):
    """
    DFS manual a partir de raiz.
    Retorna: (ordem_visita, arestas_arvore)
    """
    visitado = {n: False for n in nodes}
    ordem = []
    arestas_arvore = []

    def dfs_visit(u):
        visitado[u] = True
        ordem.append(u)
        if u in adj:
            for v, peso, edge_id, nome in adj[u]:
                if not visitado[v]:
                    arestas_arvore.append((u, v, peso, nome, edge_id))
                    dfs_visit(v)

    dfs_visit(raiz)
    return (ordem, arestas_arvore)


# ============================================================================
# CARREGAR DADOS
# ============================================================================

@st.cache_data
def carregar_dados():
    """Carrega edges.csv e nodes.csv"""
    edges_path = 'out/edges.csv'
    nodes_path = 'out/nodes.csv'

    if not os.path.exists(edges_path) or not os.path.exists(nodes_path):
        return None, None, None, None

    df_edges = pd.read_csv(edges_path)
    df_nodes = pd.read_csv(nodes_path)

    nodes = sorted(df_nodes['node'].tolist())

    # Adjacência para algoritmos: adj[u] = [(v, peso, edge_id, nome), ...]
    adj = defaultdict(list)
    for _, row in df_edges.iterrows():
        u, v = row['source'], row['target']
        peso = row['weight'] if pd.notna(row['weight']) else 1.0
        edge_id = row['edge_id']
        nome = row.get('nome_logradouro', '')
        adj[u].append((v, peso, edge_id, nome))
        adj[v].append((u, peso, edge_id, nome))

    # Degree (contagem manual)
    degree = defaultdict(int)
    for _, row in df_edges.iterrows():
        degree[row['source']] += 1
        degree[row['target']] += 1

    return df_edges, nodes, adj, degree


# ============================================================================
# VISUALIZAÇÃO COM PYVIS
# ============================================================================

# Helper global para curvatura
pair_idx = defaultdict(int)

def add_edge_curved(net, u, v, *, title=None, highlight=False, label=None):
    """Adiciona aresta com curvatura por par (evita sobreposição)"""
    global pair_idx
    a, b = (u, v) if str(u) <= str(v) else (v, u)
    i = pair_idx[(a, b)]
    pair_idx[(a, b)] += 1
    sign = 1 if (i % 2 == 0) else -1
    step = (i // 2)
    smooth = {
        "enabled": True,
        "type": "curvedCW" if sign > 0 else "curvedCCW",
        "roundness": 0.18 + 0.06 * step
    }

    base = {"smooth": smooth, "title": (title or "")}
    if highlight:
        base.update({"color": {"color": "#dc2626"}, "width": 4.0, "label": label})
    else:
        base.update({"color": {"color": "#94a3b8", "opacity": 0.45}, "width": 1.2})

    net.add_edge(u, v, **base)


def criar_grafo_visual(df_edges, nodes, degree, resultado=None, metodo=None):
    """
    Cria visualização do grafo com PyVis.

    Base: todas as 904 arestas visíveis, iguais (cinza claro, sem label).
    Destaque: conforme método (Dijkstra, BFS/DFS, Ranking).
    """
    global pair_idx
    pair_idx = defaultdict(int)  # reset

    if Network is None:
        return None

    net = Network(height="1000px", width="100%", directed=False, bgcolor="#ffffff")

    # Opções conforme especificação
    net.set_options("""{
  "layout": { "improvedLayout": true },
  "nodes": {
    "shape": "dot",
    "size": 12,
    "font": { "face": "Arial", "size": 20, "bold": false, "vadjust": 0 },
    "scaling": { "min": 20, "max": 20, "label": { "enabled": true, "min": 20, "max": 20 } },
    "labelHighlightBold": false
  },
  "edges": {
    "smooth": false,
    "color": { "color": "#94a3b8", "opacity": 0.45 },
    "width": 1.2
  },
  "physics": {
    "solver": "repulsion",
    "repulsion": { "nodeDistance": 340, "springLength": 320 },
    "stabilization": { "iterations": 450 }
  },
  "interaction": {
    "hover": true,
    "hoverConnectedEdges": true,
    "selectConnectedEdges": true,
    "zoomView": true,
    "navigationButtons": true,
    "keyboard": true,
    "tooltipDelay": 60
  }
}""")

    # Preparar conjunto de arestas destacadas
    arestas_destaque = set()
    if resultado is not None and metodo in ['dijkstra', 'bfs', 'dfs']:
        if metodo == 'dijkstra':
            _, _, caminho_arestas = resultado
            for _, _, _, _, edge_id in caminho_arestas:
                arestas_destaque.add(edge_id)
        elif metodo in ['bfs', 'dfs']:
            _, arestas_arvore = resultado
            for _, _, _, _, edge_id in arestas_arvore:
                arestas_destaque.add(edge_id)

    # Adicionar nós
    # Para Ranking: reescalar tamanho e cor por degree
    if metodo == 'ranking':
        max_deg = max(degree.values()) if degree else 1
        min_deg = min(degree.values()) if degree else 1
        for n in nodes:
            deg = degree.get(n, 0)
            # Tamanho proporcional ao grau
            size = 20 + (deg - min_deg) / (max_deg - min_deg + 1) * 30
            # Cor gradiente azul (claro → escuro)
            # Usar RGB: azul claro (173, 216, 230) → azul escuro (0, 0, 139)
            ratio = (deg - min_deg) / (max_deg - min_deg + 1)
            r = int(173 * (1 - ratio) + 0 * ratio)
            g = int(216 * (1 - ratio) + 0 * ratio)
            b = int(230 * (1 - ratio) + 139 * ratio)
            color = f"#{r:02x}{g:02x}{b:02x}"

            net.add_node(n, label=str(n), title=f"{n}\nGrau: {deg}", size=size, color=color)
    else:
        # Base: nós uniformes
        for n in nodes:
            deg = degree.get(n, 0)
            net.add_node(n, label=str(n), title=f"{n}\nGrau: {deg}")

    # Adicionar arestas
    for _, row in df_edges.iterrows():
        u, v = row['source'], row['target']
        edge_id = row['edge_id']
        nome = row.get('nome_logradouro', '')
        peso = row.get('peso', 0)
        deg_u = degree.get(u, 0)
        deg_v = degree.get(v, 0)

        # Tooltip da aresta
        tooltip = f"<b>{nome}</b><br/>{u} ↔ {v}<br/>peso: {int(peso)} m<br/>grau({u})={deg_u}  grau({v})={deg_v}"

        # Decidir se destacar
        if edge_id in arestas_destaque:
            # Destaque: vermelho, label com nome e peso
            label_aresta = f"{nome} ({int(peso)} m)" if nome else f"{int(peso)} m"
            add_edge_curved(net, u, v, title=tooltip, highlight=True, label=label_aresta)
        else:
            # Base: cinza, sem label
            add_edge_curved(net, u, v, title=tooltip, highlight=False, label=None)

    # Desligar física após estabilização para manter layout estático
    net.toggle_physics(False)

    return net


# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

def main():
    st.set_page_config(layout="wide", page_title="Grafo de Vias")

    # Carregar dados
    df_edges, nodes, adj, degree = carregar_dados()

    if df_edges is None:
        st.error("❌ Arquivos não encontrados. Execute `python criar_multigrafo_recife.py` primeiro.")
        return

    # ========== SIDEBAR ==========
    with st.sidebar:
        st.header("⚙️ Configurações")

        bairro1 = st.selectbox("Bairro 1:", options=[""] + nodes, index=0)
        bairro2 = st.selectbox("Bairro 2:", options=[""] + nodes, index=0)

        metodo = st.selectbox(
            "Método de travessia:",
            ["Dijkstra (menor caminho)", "Árvores de travessia (BFS/DFS)", "Ranking por grau (bairros)"]
        )

        executar = st.button("🔍 Executar", type="primary", use_container_width=True)

    # ========== PROCESSAMENTO ==========
    resultado = None
    metodo_id = None
    mensagem = None

    if executar:
        if metodo == "Dijkstra (menor caminho)":
            if not bairro1 or not bairro2:
                st.sidebar.error("❌ Selecione Bairro 1 e Bairro 2 para Dijkstra")
            else:
                metodo_id = 'dijkstra'
                distancia, caminho_nos, caminho_arestas = dijkstra(adj, bairro1, bairro2, nodes)
                if distancia == float('inf'):
                    st.sidebar.error(f"❌ Não há caminho entre {bairro1} e {bairro2}")
                else:
                    resultado = (distancia, caminho_nos, caminho_arestas)

                    # Mensagem: distância e vias
                    st.sidebar.success(f"✓ Distância total: **{distancia:.2f} m**")
                    st.sidebar.markdown("**Sequência de bairros:**")
                    st.sidebar.write(" → ".join(caminho_nos))

                    st.sidebar.markdown("**Vias do percurso:**")
                    for u, v, peso, nome, _ in caminho_arestas:
                        st.sidebar.write(f"- {nome} ({int(peso)} m)")

        elif metodo == "Árvores de travessia (BFS/DFS)":
            if not bairro1:
                st.sidebar.error("❌ Selecione Bairro 1 como raiz")
            else:
                # Escolher BFS ou DFS (pode adicionar um sub-select ou alternar)
                # Por simplicidade, vamos usar BFS. Se quiser escolha, adicione outro selectbox
                escolha_busca = st.sidebar.radio("Tipo de busca:", ["BFS (Largura)", "DFS (Profundidade)"])

                if escolha_busca == "BFS (Largura)":
                    metodo_id = 'bfs'
                    ordem, arestas_arvore = bfs_tree(adj, bairro1, nodes)
                else:
                    metodo_id = 'dfs'
                    ordem, arestas_arvore = dfs_tree(adj, bairro1, nodes)

                resultado = (ordem, arestas_arvore)

                st.sidebar.success(f"✓ Árvore de travessia ({escolha_busca}) a partir de **{bairro1}**")
                st.sidebar.markdown(f"**Nós visitados:** {len(ordem)}")
                st.sidebar.markdown("**Ordem de visita:**")
                st.sidebar.write(", ".join(ordem[:20]) + ("..." if len(ordem) > 20 else ""))

        elif metodo == "Ranking por grau (bairros)":
            metodo_id = 'ranking'
            # Ordenar nós por grau decrescente
            ranking = sorted(degree.items(), key=lambda x: x[1], reverse=True)
            resultado = ranking  # apenas para sinalizar que tem resultado

            st.sidebar.success("✓ Ranking por grau")
            st.sidebar.markdown("**Top 20 bairros por grau:**")
            for i, (bairro, deg) in enumerate(ranking[:20], 1):
                st.sidebar.write(f"{i}. **{bairro}**: {deg} conexões")

    # ========== VISUALIZAÇÃO ==========
    net = criar_grafo_visual(df_edges, nodes, degree, resultado, metodo_id)

    if net is not None:
        html = net.generate_html()
        st.components.v1.html(html, height=1000, scrolling=True)


if __name__ == '__main__':
    main()
