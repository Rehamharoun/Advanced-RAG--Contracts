"""Lightweight contracts knowledge graph (entities, clauses, documents).

Used to boost documents whose entities appear in the user query.
"""
import re
from collections import defaultdict
from itertools import combinations

import networkx as nx

# Aliases are merged into one node (simple entity resolution)
ENTITIES = {
    "Juhayna": ("Organization", ["juhayna food industries", "juhayna"]),
    "PepsiCo Egypt": ("Organization", ["pepsico egypt for trading and distribution", "pepsico egypt", "pepsico"]),
    "Vodafone Egypt": ("Organization", ["vodafone egypt telecommunications", "vodafone egypt", "vodafone"]),
    "Delta Cold Logistics": ("Organization", ["delta cold logistics"]),
    "Nile Digital Solutions": ("Organization", ["nile digital solutions", "nile digital"]),
    "Meta Platforms": ("Organization", ["meta platforms", "meta"]),
    "Client Corp": ("Organization", ["client corp"]),
    "Provider LLC": ("Organization", ["provider llc"]),
    "WhatsApp": ("Product", ["whatsapp"]),
    "Facebook": ("Product", ["facebook"]),
}

CLAUSES = {
    "Liability Cap": ["limitation of liability", "aggregate liability", "total liability"],
    "Termination": ["terminat"],
    "Confidentiality": ["confidential"],
    "Data Protection": ["data protection", "personal data", "privacy"],
    "Governing Law": ["governing law", "governed by"],
    "Payment Terms": ["payment", "invoice"],
    "Intellectual Property": ["intellectual property"],
}

ENTITY_PATTERNS = {
    name: re.compile(r"\b(" + "|".join(re.escape(a) for a in aliases) + r")\b", re.I)
    for name, (_, aliases) in ENTITIES.items()
}


def build_graph(docs: dict[str, str]) -> nx.Graph:
    G = nx.Graph()
    for name, (typ, _) in ENTITIES.items():
        G.add_node(name, type=typ)
    for c in CLAUSES:
        G.add_node(c, type="Clause")

    G.add_edge("Meta Platforms", "WhatsApp", relation="owns")
    G.add_edge("Meta Platforms", "Facebook", relation="owns")

    for doc_name, text in docs.items():
        G.add_node(doc_name, type="Document")
        low = text.lower()
        head = low[:1500]
        parties = []

        for ent, pat in ENTITY_PATTERNS.items():
            n = len(pat.findall(low))
            if n >= 2:
                G.add_edge(ent, doc_name, relation="mentioned_in", count=n)
            if pat.search(head) and ENTITIES[ent][0] == "Organization":
                parties.append(ent)

        for clause, words in CLAUSES.items():
            if sum(low.count(w) for w in words) >= 2:
                G.add_edge(doc_name, clause, relation="has_clause")

        for a, b in combinations(parties, 2):
            if not G.has_edge(a, b):
                G.add_edge(a, b, relation="contract_with", docs=[])
            G.edges[a, b].setdefault("docs", []).append(doc_name)

    print("graph:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")
    for a, b, d in G.edges(data=True):
        if d["relation"] == "contract_with":
            print(f"  {a} <-> {b}  {d['docs']}")
    return G


def docs_for_query(G: nx.Graph, query: str) -> dict[str, int]:
    """Documents linked to entities mentioned in the query -> boost weight."""
    hits: dict[str, int] = defaultdict(int)
    for ent, pat in ENTITY_PATTERNS.items():
        if pat.search(query):
            for nb in G.neighbors(ent):
                if G.nodes[nb]["type"] == "Document":
                    hits[nb] += 1
    return hits


def draw_graph(G: nx.Graph, out_path: str = "outputs/knowledge_graph.png") -> None:
    import os

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {"Organization": "#4C9BE8", "Product": "#F2A541"}
    H = G.edge_subgraph([(a, b) for a, b, d in G.edges(data=True)
                         if d["relation"] in ("contract_with", "owns")]).copy()
    node_colors = [colors.get(H.nodes[n].get("type"), "#cccccc") for n in H.nodes]

    plt.figure(figsize=(14, 9))
    pos = nx.spring_layout(H, k=1.2, seed=42)
    nx.draw_networkx_edges(H, pos, width=2.5, alpha=0.8)
    nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=2500)
    nx.draw_networkx_labels(H, pos, font_size=9)

    labels = {}
    for a, b, d in H.edges(data=True):
        labels[(a, b)] = ("contract\n" + ", ".join(x[:2] for x in d["docs"])
                          if d["relation"] == "contract_with" else d["relation"])
    nx.draw_networkx_edge_labels(H, pos, edge_labels=labels, font_size=8)

    legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=12, label=t)
              for t, c in colors.items()]
    plt.legend(handles=legend, loc="upper left")
    plt.title("Contracts Knowledge Graph (parties and contracts)")
    plt.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
