#!/usr/bin/env python3
"""
App Streamlit para visualização e análise de grafo de vias de Recife
Implementa Dijkstra e Kruskal conforme slides das aulas (sem algoritmos prontos)
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
from collections import defaultdict
import heapq

try:
    from pyvis.network import Network
except ImportError:
    Network = None


# ============================================================================
# UNIÃO-ENCONTRO (Union-Find) para Kruskal
# Referência: AULA 12 - AMG_KRUSKAL
# ============================================================================

class UnionFind:
    """
    Union-Find com path compression e union by rank.
    Usado no Kruskal para detectar ciclos e unir componentes.
    """
    def __init__(self, nodes):
        self.parent = {n: n for n in nodes}
        self.rank = {n: 0 for n in nodes}

    def find(self, x):
        """Encontra a raiz do conjunto com path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x, y):
        """Une dois conjuntos usando union by rank. Retorna True se uniu, False se já estavam unidos."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # já estão no mesmo conjunto (evita ciclo)

        # union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True


# ============================================================================
# DIJKSTRA - Caminho mais curto de s → t
# Referência: AULA 05 - CAMINHOS MAIS CURTO_DIJKSTRA
# ============================================================================

def dijkstra(adj, s, t, nodes):
    """
    Implementação do algoritmo de Dijkstra conforme slides da aula.

    Vetores:
    - d[v]: estimativa de distância de s até v
    - p[v]: predecessor de v no caminho
    - aberto[v]: se v ainda está aberto (não processado)

    Args:
        adj: dicionário {u: [(v, peso, edge_id, nome), ...]}
        s: vértice origem
        t: vértice destino
        nodes: lista de todos os nós

    Returns:
        (distancia, caminho_nos, caminho_arestas)
        caminho_arestas = [(u, v, peso, nome), ...]
    """
    # Inicialização (slide da aula)
    INF = float('inf')
    d = {v: INF for v in nodes}  # distância estimada
    p = {v: None for v in nodes}  # predecessor
    p_edge = {v: None for v in nodes}  # aresta que levou até v (edge_id, nome, peso)
    aberto = {v: True for v in nodes}  # vértices ainda não processados

    d[s] = 0

    # Fila de prioridade (min-heap): (distância, vértice)
    heap = [(0, s)]

    # Loop principal do Dijkstra
    while heap:
        dist_u, u = heapq.heappop(heap)

        # Se já foi processado (fechado), ignora
        if not aberto[u]:
            continue

        # Fecha u (marca como processado)
        aberto[u] = False

        # Se chegou no destino, pode parar
        if u == t:
            break

        # Relaxamento: para cada vizinho v de u
        if u in adj:
            for v, peso, edge_id, nome in adj[u]:
                if aberto[v]:  # só relaxa se v ainda estiver aberto
                    # Tenta melhorar a estimativa de d[v]
                    nova_dist = d[u] + peso
                    if nova_dist < d[v]:
                        d[v] = nova_dist
                        p[v] = u
                        p_edge[v] = (edge_id, nome, peso, u)
                        heapq.heappush(heap, (nova_dist, v))

    # Reconstruir caminho de s → t usando p[v]
    if d[t] == INF:
        return (INF, [], [])  # não há caminho

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
# KRUSKAL - Árvore Geradora Mínima (AGM)
# Referência: AULA 12 - AMG_KRUSKAL, AULA 11 - AMG
# ============================================================================

def kruskal(edges_list, nodes):
    """
    Implementação do algoritmo de Kruskal conforme slides da aula.

    Passos:
    1. Criar uma floresta (cada vértice é uma árvore)
    2. Ordenar arestas por peso crescente
    3. Para cada aresta (u,v):
       - Se u e v estão em árvores diferentes, une e adiciona aresta à AGM
       - Caso contrário, pula (evita ciclo)
    4. AGM final tem |V|-1 arestas

    Args:
        edges_list: [(peso, edge_id, u, v, nome), ...]
        nodes: lista de todos os nós

    Returns:
        (custo_total, agm_arestas)
        agm_arestas = [(u, v, peso, nome, edge_id), ...]
    """
    # 1. Criar floresta (Union-Find)
    uf = UnionFind(nodes)

    # 2. Ordenar arestas por peso crescente
    edges_sorted = sorted(edges_list, key=lambda e: e[0])

    # 3. Processar arestas em ordem
    agm_arestas = []
    custo_total = 0.0

    for peso, edge_id, u, v, nome in edges_sorted:
        # Tenta unir u e v
        if uf.union(u, v):
            # Estavam em componentes diferentes → adiciona aresta à AGM
            agm_arestas.append((u, v, peso, nome, edge_id))
            custo_total += peso

            # Quando tiver |V|-1 arestas, a AGM está completa
            if len(agm_arestas) == len(nodes) - 1:
                break

    return (custo_total, agm_arestas)


# ============================================================================
# CARREGAR DADOS (edges.csv e nodes.csv)
# ============================================================================

@st.cache_data
def carregar_dados():
    """Carrega edges.csv e nodes.csv gerados por criar_multigrafo_recife.py"""
    edges_path = 'out/edges.csv'
    nodes_path = 'out/nodes.csv'

    if not os.path.exists(edges_path) or not os.path.exists(nodes_path):
        return None, None, None, None, None

    df_edges = pd.read_csv(edges_path)
    df_nodes = pd.read_csv(nodes_path)

    # Lista de nós
    nodes = sorted(df_nodes['node'].tolist())

    # Lista de arestas para Kruskal: (peso, edge_id, u, v, nome)
    edges_list = []
    for _, row in df_edges.iterrows():
        peso = row['weight'] if pd.notna(row['weight']) else 1.0
        edges_list.append((
            peso,
            row['edge_id'],
            row['source'],
            row['target'],
            row.get('nome_logradouro', '')
        ))

    # Adjacência para Dijkstra: adj[u] = [(v, peso, edge_id, nome), ...]
    adj = defaultdict(list)
    for _, row in df_edges.iterrows():
        u = row['source']
        v = row['target']
        peso = row['weight'] if pd.notna(row['weight']) else 1.0
        edge_id = row['edge_id']
        nome = row.get('nome_logradouro', '')

        # Grafo não dirigido: adiciona ambas as direções
        adj[u].append((v, peso, edge_id, nome))
        adj[v].append((u, peso, edge_id, nome))

    return df_edges, df_nodes, nodes, edges_list, adj


# ============================================================================
# VISUALIZAÇÃO COM PYVIS
# ============================================================================

def criar_visualizacao_base(df_edges, nodes, resultado=None, tipo_algo=None):
    """
    Cria visualização base (mapa) com PyVis.

    Visual base:
    - Arestas todas iguais (sem label, espessura e cor uniformes)
    - Nós com label sempre visível
    - Multiarestas curvadas (roundness incremental, alternando CW/CCW)

    Destaque de resultado:
    - Dijkstra: destaca arestas do caminho com cor azul e espessura maior
    - Kruskal: destaca arestas da AGM com cor azul e espessura maior
    """
    if Network is None:
        st.error("PyVis não instalado. Execute: pip install pyvis")
        return None

    net = Network(height="750px", width="100%", directed=False, bgcolor="#ffffff")

    # Configurações conforme especificação
    net.set_options("""{
  "nodes": {
    "shape": "dot",
    "font": { "size": 18, "face": "Arial", "vadjust": 0 },
    "scaling": { "min": 16, "max": 16, "label": { "enabled": true, "min": 16, "max": 16 } },
    "labelHighlightBold": false
  },
  "edges": {
    "smooth": false,
    "color": { "color": "#94a3b8", "opacity": 0.65 },
    "width": 1.5
  },
  "physics": {
    "solver": "repulsion",
    "repulsion": { "nodeDistance": 240, "springLength": 240 },
    "stabilization": { "iterations": 250 }
  },
  "interaction": { "hover": true, "zoomView": true }
}""")

    # Adicionar nós (label sempre visível)
    for n in nodes:
        net.add_node(n, label=str(n), title=str(n))

    # Preparar conjunto de arestas destacadas (se houver resultado)
    arestas_destaque = set()
    if resultado is not None and tipo_algo in ['dijkstra', 'kruskal']:
        if tipo_algo == 'dijkstra':
            _, _, caminho_arestas = resultado
            for u, v, peso, nome, edge_id in caminho_arestas:
                arestas_destaque.add(edge_id)
        elif tipo_algo == 'kruskal':
            _, agm_arestas = resultado
            for u, v, peso, nome, edge_id in agm_arestas:
                arestas_destaque.add(edge_id)

    # Adicionar arestas com curvatura por par
    pair_idx = defaultdict(int)

    for _, row in df_edges.iterrows():
        u = row['source']
        v = row['target']
        edge_id = row['edge_id']
        nome = row.get('nome_logradouro', '')
        peso = row.get('peso', 0)

        # Normaliza o par (não dirigido)
        a, b = (u, v) if str(u) <= str(v) else (v, u)
        i = pair_idx[(a, b)]
        pair_idx[(a, b)] += 1

        # Alterna sentido e aumenta curvatura
        sign = 1 if (i % 2 == 0) else -1
        step = (i // 2)
        roundness = 0.18 + 0.06 * step

        smooth = {
            "enabled": True,
            "type": "curvedCW" if sign > 0 else "curvedCCW",
            "roundness": roundness
        }

        # Estilo da aresta (base ou destaque)
        if edge_id in arestas_destaque:
            # Destaque: cor azul, espessura maior, label com nome e peso
            label_aresta = f"{nome} ({int(peso)} m)" if nome else f"{int(peso)} m"
            net.add_edge(
                u, v,
                title=label_aresta,
                label=label_aresta,
                color="#2563eb",
                width=4.0,
                smooth=smooth
            )
        else:
            # Base: sem label, estilo padrão
            net.add_edge(
                u, v,
                title="",
                color={"color": "#94a3b8", "opacity": 0.65},
                width=1.5,
                smooth=smooth
            )

    return net


# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

def main():
    st.set_page_config(page_title="Grafo de Vias - Recife", layout="wide")

    st.title("📍 Grafo de Vias de Recife")
    st.markdown("**Visualização e análise com Dijkstra e Kruskal (implementados conforme slides das aulas)**")

    # Instruções de uso
    with st.expander("ℹ️ Como usar"):
        st.markdown("""
        **Instalação:**
        ```bash
        pip install streamlit pyvis pandas openpyxl
        streamlit run app.py
        ```

        **Algoritmos implementados:**
        - **Dijkstra**: Caminho mais curto entre dois bairros (AULA 05)
        - **Kruskal**: Árvore Geradora Mínima - AGM (AULA 11, 12)

        **Dados:**
        - Use `criar_multigrafo_recife.py` para gerar os arquivos CSV/GEXF
        - O app carrega `out/edges.csv` e `out/nodes.csv`
        """)

    # Carregar dados
    df_edges, df_nodes, nodes, edges_list, adj = carregar_dados()

    if df_edges is None:
        st.error("❌ Arquivos não encontrados. Execute `python criar_multigrafo_recife.py` primeiro.")
        return

    st.success(f"✓ Dados carregados: **{len(nodes)} nós**, **{len(df_edges)} arestas** (904 esperadas)")

    # Sidebar: seleção de algoritmo
    st.sidebar.header("⚙️ Configurações")

    modo = st.sidebar.selectbox(
        "Algoritmo:",
        ["Viz base (nenhum)", "Dijkstra (menor caminho)", "AGM (Kruskal)"]
    )

    # Variáveis de resultado
    resultado = None
    tipo_algo = None

    # ========== DIJKSTRA ==========
    if modo == "Dijkstra (menor caminho)":
        st.sidebar.subheader("Dijkstra - Caminho mais curto")

        origem = st.sidebar.selectbox("Origem:", nodes, index=0)
        destino = st.sidebar.selectbox("Destino:", nodes, index=1 if len(nodes) > 1 else 0)

        if st.sidebar.button("🔍 Executar Dijkstra"):
            with st.spinner("Calculando caminho mais curto..."):
                distancia, caminho_nos, caminho_arestas = dijkstra(adj, origem, destino, nodes)
                resultado = (distancia, caminho_nos, caminho_arestas)
                tipo_algo = 'dijkstra'

                if distancia == float('inf'):
                    st.error(f"❌ Não há caminho entre **{origem}** e **{destino}**")
                else:
                    st.success(f"✓ Caminho encontrado! Distância total: **{distancia:.2f} m**")

                    st.subheader("📍 Sequência de bairros:")
                    st.write(" → ".join(caminho_nos))

                    st.subheader("🛣️ Arestas do caminho:")
                    for u, v, peso, nome, edge_id in caminho_arestas:
                        st.write(f"- **{u}** → **{v}**: {nome} ({peso:.2f} m)")

    # ========== KRUSKAL ==========
    elif modo == "AGM (Kruskal)":
        st.sidebar.subheader("Kruskal - Árvore Geradora Mínima")

        if st.sidebar.button("🌳 Executar Kruskal"):
            with st.spinner("Calculando AGM..."):
                custo_total, agm_arestas = kruskal(edges_list, nodes)
                resultado = (custo_total, agm_arestas)
                tipo_algo = 'kruskal'

                st.success(f"✓ AGM construída! Custo total: **{custo_total:.2f} m**")
                st.info(f"Arestas na AGM: **{len(agm_arestas)}** (esperado: {len(nodes)-1} = |V|-1)")

                st.subheader("🌳 Arestas da AGM:")
                for u, v, peso, nome, edge_id in agm_arestas[:20]:  # mostra primeiras 20
                    st.write(f"- **{u}** ↔ **{v}**: {nome} ({peso:.2f} m)")

                if len(agm_arestas) > 20:
                    st.write(f"... e mais {len(agm_arestas) - 20} arestas")

    # ========== VISUALIZAÇÃO ==========
    st.subheader("🗺️ Visualização do Grafo")

    with st.spinner("Gerando visualização..."):
        net = criar_visualizacao_base(df_edges, nodes, resultado, tipo_algo)

        if net is not None:
            # Salvar HTML temporário
            html_path = 'out/grafo_interativo_temp.html'
            Path('out').mkdir(exist_ok=True)
            net.save_graph(html_path)

            # Exibir HTML no Streamlit
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            st.components.v1.html(html_content, height=800, scrolling=True)

            # Botão de export
            if st.sidebar.button("💾 Salvar visualização"):
                net.save_graph('out/grafo_interativo.html')
                st.sidebar.success("✓ Salvo em `out/grafo_interativo.html`")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Grafo de Vias de Recife**")
    st.sidebar.markdown("88 bairros | 904 vias")


if __name__ == '__main__':
    main()
