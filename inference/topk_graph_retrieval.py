import torch
import networkx as nx
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
from typing import List, Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'PE-Core-G14-448'
model = pe.CLIP.from_config(model_name, pretrained=True).to(device)
tokenizer = transforms.get_text_tokenizer(model.context_length)


def _encode_query(query: str) -> torch.Tensor:
    text = tokenizer([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.squeeze(0)


def retrieve_topk_segments(
    graph: nx.Graph,
    query: str,
    top_k: int = 3,
    spatial_k: int = 3,
) -> List[Dict]:
    """根据文本查询在图中检索最相关的片段。

    Args:
        graph: 构建好的时空图。
        query: 检索文本。
        top_k: 选取的最相似节点数量。
        spatial_k: 每个 top 节点额外扩展的空间邻居数量。

    Returns:
        list[dict]: 包含 ``node_id``、``similarity`` 以及节点元信息的列表。
    """

    if graph.number_of_nodes() == 0:
        return []

    text_feat = _encode_query(query)

    similarities = []
    for node_id, attrs in graph.nodes(data=True):
        img_feat = attrs["feature"].to(device)
        img_feat = img_feat / img_feat.norm()
        sim = torch.dot(img_feat, text_feat).item()
        similarities.append((node_id, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_nodes = similarities[: max(top_k, 1)]

    selected_nodes = set(nid for nid, _ in top_nodes)

    for nid, _ in top_nodes:
        spatial_neighbors = [
            (nbr, graph.edges[nid, nbr]["weight"])
            for nbr in graph.neighbors(nid)
            if graph.edges[nid, nbr]["type"] == "spatial"
        ]
        spatial_neighbors.sort(key=lambda x: x[1], reverse=True)
        selected_nodes.update(nbr for nbr, _ in spatial_neighbors[:spatial_k])

    for nid in list(selected_nodes):
        for nbr in graph.neighbors(nid):
            if graph.edges[nid, nbr]["type"] == "temporal":
                selected_nodes.add(nbr)

    ordered_nodes = sorted(selected_nodes)
    node_lookup = dict(similarities)

    results: List[Dict] = []
    for nid in ordered_nodes:
        info = dict(graph.nodes[nid])
        info["node_id"] = nid
        info["similarity"] = node_lookup.get(nid)
        results.append(info)

    return results


if __name__ == "__main__":
    import argparse
    import pickle
    import json

    parser = argparse.ArgumentParser(description="Retrieve top-k segments from a spatio-temporal graph")
    parser.add_argument("graph", help="Path to the pickled graph")
    parser.add_argument("query", help="Text query")
    parser.add_argument("--top-k", type=int, default=1, dest="top_k")
    parser.add_argument("--spatial-k", type=int, default=3, dest="spatial_k")
    parser.add_argument("--output", help="Optional path to save selected segments as JSON")

    args = parser.parse_args()

    with open(args.graph, "rb") as f:
        graph = pickle.load(f)

    results = retrieve_topk_segments(graph, args.query, top_k=args.top_k, spatial_k=args.spatial_k)

    print(json.dumps(results, ensure_ascii=False, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved results to {args.output}")
