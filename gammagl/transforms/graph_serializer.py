import operator
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import tensorlayerx as tlx
import numpy as np

from gammagl.data import Graph


SERIALIZER_PROTOCOL_VERSION = 5
OFFICIAL_REVISION = "98343a6b025a48fbb6859cd812a12b81ec3ac3cc"


def validate_undirected_graph_semantics(graph: Any) -> None:
    """Reject explicit direction metadata, ignoring GammaGL's COO inference."""
    if isinstance(graph, Graph):
        # BaseGraph.is_directed() infers direction from COO symmetry, and a
        # stored is_directed field is shadowed by that method on attribute lookup.
        declarations = graph.to_dict()
    elif isinstance(graph, dict):
        declarations = graph
    else:
        declarations = {name: getattr(graph, name, None)
                        for name in ("directed", "is_directed")}
    for name in ("directed", "is_directed"):
        value = declarations.get(name)
        if callable(value):
            value = value()
        if value is not None and bool(value):
            raise ValueError(
                "Feuler only supports undirected simple graphs; "
                f"directed graph semantics are unsupported ({name}=True).")


@dataclass
class GraphSerializationResult:
    """Container for serialized graph tokens and lightweight metadata."""

    token_ids: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)


class FrequencyGuidedEulerianSerializer:
    """Feuler for simple undirected graphs, with two explicit input protocols.

    ``official`` follows the pinned source's traversal and loader token rules.
    It has no node identity tokens and cannot reconstruct a graph. ``local_refs``
    is the optional reversible GammaGL protocol described below.

    ``edge_index`` may contain one or both directions of an undirected edge,
    but every direction must carry the same edge label. Directed graphs,
    parallel edges, and self-loops are not representable by this serializer.

    The ``local_refs`` grammar is self-describing: a component starts with a node-label
    token and a node-reference token, then contains repeated
    ``edge-label, node-label, node-reference`` triples. Components are joined
    by ``component_sep_token_id``. Node labels, edge labels, and node
    references occupy disjoint integer domains. The stream recovers an
    isomorphic labeled graph; result metadata restores original node IDs.
    Graphs without ``x`` use zero as every node label, including in the
    reconstructed ``x``; absent original features cannot be recovered.
    """

    def __init__(
        self,
        name: str = "feuler",
        include_edge_tokens: bool = True,
        component_sep_token_id: Optional[int] = None,
        protocol: str = "official",
        dataset: Optional[str] = None,
    ):
        self.name = name
        if protocol not in ("official", "local_refs"):
            raise ValueError("protocol must be official or local_refs.")
        self.protocol = protocol
        self.dataset = dataset
        self.include_edge_tokens = include_edge_tokens
        if component_sep_token_id is None:
            component_sep_token_id = 7 if protocol == "official" else -1
        self.component_sep_token_id = self._strict_integer(
            component_sep_token_id,
            "component_sep_token_id must be an integer.",
        )
        if protocol == "local_refs" and self.component_sep_token_id >= 0:
            raise ValueError("component_sep_token_id must be a negative integer.")
        self.frequency_map: Dict[Any, int] = {}

    def configuration(self):
        return dict(method=self.name, protocol=self.protocol, dataset=self.dataset,
                    include_edge_tokens=self.include_edge_tokens,
                    component_sep_token_id=self.component_sep_token_id,
                    official_revision=OFFICIAL_REVISION)

    def fit(self, graphs: Sequence[Any]):
        counts: Dict[Tuple[int, int, int], int] = defaultdict(int)
        for graph in graphs:
            graph_view = self._read_graph(graph)
            edges = (graph_view["input_edges"] if self.protocol == "official"
                     else graph_view["canonical_edges"])
            for src, dst, edge_label in edges:
                pattern = ((graph_view["node_labels"][src], edge_label,
                            graph_view["node_labels"][dst]) if self.protocol == "official"
                           else self._edge_pattern(graph_view, src, dst, edge_label))
                counts[pattern] += 1
        self.frequency_map = dict(counts)
        return self

    def serialize(
            self, graph: Any, start_node: Optional[int] = None
    ) -> GraphSerializationResult:
        graph_view = self._read_graph(graph)
        if start_node is not None:
            start_node = int(start_node)
            if not 0 <= start_node < graph_view["num_nodes"]:
                raise ValueError("start_node must identify a node in the graph.")
        if self.protocol == "official":
            return self._serialize_official(graph_view, start_node)
        components = self._connected_components(graph_view["num_nodes"], graph_view["arcs"])
        component_results = []

        for component in components:
            component_start = start_node if start_node in component else None
            edges = self._serialize_component(
                graph_view, component, start_node=component_start)
            component_results.append((
                edges,
                component,
                component_start,
                self._traversal_signature(graph_view, edges, component),
            ))

        component_results.sort(key=lambda item: (-len(item[0]), item[3]))
        token_ids: List[int] = []
        local_references: Dict[int, int] = {}
        for index, (edges, component, component_start, _) in enumerate(component_results):
            if index > 0:
                token_ids.append(self.component_sep_token_id)
            token_ids.extend(self._edges_to_tokens(
                graph_view, edges, component, local_references,
                start_node=component_start))

        return GraphSerializationResult(
            token_ids=token_ids,
            metadata={
                "method": self.name,
                "protocol_version": SERIALIZER_PROTOCOL_VERSION,
                "protocol": self.protocol,
                "num_nodes": graph_view["num_nodes"],
                "num_components": len(component_results),
                "num_edges_traversed": sum(
                    len(edges) for edges, _, _, _ in component_results),
                "start_node": start_node,
                "local_to_original": list(local_references),
            },
        )

    def _serialize_official(self, view, start_node):
        """Port of the pinned Eulerian/Feuler and base path-to-token methods.

        Preserve input COO arcs: the official DGL implementation does not
        synthesize reverse arcs for Feuler. Eulerian sorts IDs ascending and
        pops the largest; Feuler sorts by descending directional triplet count,
        then ascending neighbor ID. Disconnected starts are reset per component.
        """
        if view["num_nodes"] == 0:
            raise ValueError("Graph has no nodes, cannot serialize.")
        components = self._connected_components(view["num_nodes"], view["input_edges"])
        components.sort(key=len, reverse=True)
        tokens, paths = [], []
        classification_loader = self.dataset in ("mutagenicity", "dblp")
        for component in components:
            # Match NetworkX component sets passed to dgl.node_subgraph.
            nodes = list(set(component))
            local_id = {node: i for i, node in enumerate(nodes)}
            start = (0 if start_node is None else start_node) if len(components) == 1 else nodes[0]
            adjacency = {node: [] for node in nodes}
            edge_labels = {}
            for src, dst, label in view["input_edges"]:
                if src in local_id and dst in local_id:
                    adjacency[src].append(dst)
                    edge_labels[src, dst] = label
            if self.name == "eulerian":
                for neighbors in adjacency.values():
                    neighbors.sort(key=local_id.get)
                degrees = dict.fromkeys(nodes, 0)
                for src, neighbors in adjacency.items():
                    degrees[src] += len(neighbors)
                    for dst in neighbors:
                        degrees[dst] += 1
                if any(degree % 2 for degree in degrees.values()):
                    doubled = {node: [] for node in nodes}
                    for src in nodes:
                        for dst in adjacency[src]:
                            doubled[src].append(dst)
                            doubled[dst].append(src)
                    adjacency = doubled
            else:
                for src, neighbors in adjacency.items():
                    neighbors.sort(key=lambda dst: (
                        -float(np.log10(np.float32(max(1, self.frequency_map.get((view["node_labels"][src],
                             edge_labels[src, dst], view["node_labels"][dst]), 0))))), local_id[dst]))
                    neighbors.reverse()
                if not adjacency[start] and len(nodes) > 1:
                    start = next(node for node in nodes if adjacency[node])
            stack, path = [start], []
            while stack:
                current = stack[-1]
                if adjacency[current]:
                    stack.append(adjacency[current].pop())
                else:
                    path.append(stack.pop())
            path.reverse()
            if tokens:
                tokens.append(self.component_sep_token_id)  # ordinary official BPE alphabet
            for i, node in enumerate(path):
                # Official TU loaders differ between their scalar singleton
                # getter and whole-graph getter; preserve this observable rule.
                label = view["node_labels"][node]
                tokens.append(2 * label + 1 if classification_loader and len(nodes) > 1 else label)
                if i + 1 < len(path) and self.include_edge_tokens:
                    key = node, path[i + 1]
                    if key not in edge_labels:
                        raise ValueError(f"Official traversal requires an input COO edge {key}.")
                    edge = edge_labels[key]
                    omitted = 2 if self.dataset == "aqsol" else 0
                    if edge != omitted:
                        tokens.append(2 * edge if classification_loader else edge)
            paths.append(path)
        return GraphSerializationResult(tokens, dict(
            method=self.name, protocol=self.protocol, protocol_version=SERIALIZER_PROTOCOL_VERSION,
            official_revision=OFFICIAL_REVISION, num_nodes=view["num_nodes"],
            num_components=len(components), start_node=start_node,
            num_edges_traversed=sum(len(path) - 1 for path in paths)))

    def restore_input_metadata(self, result: GraphSerializationResult) -> Dict[str, List[int]]:
        """Restore original node IDs from a full result and its provenance map."""
        if not isinstance(result, GraphSerializationResult):
            raise ValueError("Original node IDs require a full serializer result with metadata.")
        local = self.deserialize(result)
        mapping = result.metadata.get("local_to_original")
        if not isinstance(mapping, list) or sorted(mapping) != list(range(local["num_nodes"])):
            raise ValueError("Serializer result has invalid local_to_original metadata.")
        canonical_edges = sorted(
            (min(mapping[src], mapping[dst]), max(mapping[src], mapping[dst]), label)
            for src, dst, label in zip(
                local["edge_index"][0], local["edge_index"][1], local["edge_attr"]))
        labels = [0] * local["num_nodes"]
        for local_id, original_id in enumerate(mapping):
            labels[original_id] = local["x"][local_id]
        return {
            "edge_index": [[src for src, _, _ in canonical_edges],
                           [dst for _, dst, _ in canonical_edges]],
            "x": labels,
            "edge_attr": [label for _, _, label in canonical_edges],
            "num_nodes": local["num_nodes"],
        }

    def deserialize(self, result: GraphSerializationResult) -> Dict[str, List[int]]:
        """Reconstruct a graph from a Feuler token stream without metadata."""
        if self.protocol == "official":
            raise NotImplementedError(
                "Official tokens omit node identity and common edge tokens; cannot reconstruct a graph.")
        if not self.include_edge_tokens:
            raise ValueError("Feuler deserialization requires include_edge_tokens=True.")
        if isinstance(result, GraphSerializationResult):
            if result.metadata.get("protocol_version") != SERIALIZER_PROTOCOL_VERSION:
                raise ValueError("Incompatible serializer protocol version; rebuild the result.")
            if result.metadata.get("protocol") != self.protocol or result.metadata.get("method") != self.name:
                raise ValueError("Incompatible serializer protocol or method.")
            token_ids = result.token_ids
        else:
            token_ids = result
        components = self._split_components(token_ids)
        labels_by_node: Dict[int, int] = {}
        edges_by_key: Dict[Tuple[int, int], int] = {}

        for component in components:
            if not component:
                raise ValueError("Feuler token stream contains an empty component.")
            if len(component) < 2:
                raise ValueError("Feuler component must begin with a node label and reference.")
            current, label = self._parse_node(component, 0)
            self._record_node_label(labels_by_node, current, label)
            index = 2
            while index < len(component):
                if index + 2 >= len(component):
                    raise ValueError("Feuler edge traversal must contain edge and node tokens.")
                edge_label = self._decode_edge_token(component[index])
                nxt, label = self._parse_node(component, index + 1)
                self._record_node_label(labels_by_node, nxt, label)
                if current == nxt:
                    raise ValueError("Feuler token stream contains an unsupported self-loop.")
                key = (min(current, nxt), max(current, nxt))
                previous = edges_by_key.setdefault(key, edge_label)
                if previous != edge_label:
                    raise ValueError(
                        f"Feuler token stream assigns conflicting labels to edge {key}.")
                current = nxt
                index += 3

        if not labels_by_node:
            return {"edge_index": [[], []], "x": [], "edge_attr": [], "num_nodes": 0}
        num_nodes = max(labels_by_node) + 1
        expected_nodes = set(range(num_nodes))
        if set(labels_by_node) != expected_nodes:
            raise ValueError("Feuler node references must be contiguous from zero.")
        canonical_edges = [(src, dst, label) for (src, dst), label in sorted(edges_by_key.items())]
        return {
            "edge_index": [
                [src for src, _, _ in canonical_edges],
                [dst for _, dst, _ in canonical_edges],
            ],
            "x": [labels_by_node[node] for node in range(num_nodes)],
            "edge_attr": [label for _, _, label in canonical_edges],
            "num_nodes": num_nodes,
        }

    @staticmethod
    def node_token(label: int) -> int:
        return 3 * FrequencyGuidedEulerianSerializer._encode_integer(label)

    @staticmethod
    def edge_token(label: int) -> int:
        return 3 * FrequencyGuidedEulerianSerializer._encode_integer(label) + 1

    @staticmethod
    def node_reference_token(node_id: int) -> int:
        if int(node_id) < 0:
            raise ValueError("Feuler node references must be non-negative.")
        return 3 * int(node_id) + 2

    @staticmethod
    def _encode_integer(value: int) -> int:
        value = int(value)
        return 2 * value if value >= 0 else -2 * value - 1

    @staticmethod
    def _decode_integer(value: int) -> int:
        return value // 2 if value % 2 == 0 else -(value // 2) - 1

    def _parse_node(self, tokens: Sequence[int], index: int) -> Tuple[int, int]:
        return self._decode_node_reference_token(tokens[index + 1]), self._decode_node_token(tokens[index])

    def _decode_node_token(self, token: int) -> int:
        token = int(token)
        if token < 0 or token % 3 != 0:
            raise ValueError(f"Expected a Feuler node-label token, got {token}.")
        return self._decode_integer(token // 3)

    def _decode_edge_token(self, token: int) -> int:
        token = int(token)
        if token < 0 or token % 3 != 1:
            raise ValueError(f"Expected a Feuler edge-label token, got {token}.")
        return self._decode_integer(token // 3)

    @staticmethod
    def _decode_node_reference_token(token: int) -> int:
        token = int(token)
        if token < 0 or token % 3 != 2:
            raise ValueError(f"Expected a Feuler node-reference token, got {token}.")
        return token // 3

    def _split_components(self, token_ids: Sequence[int]) -> List[List[int]]:
        components = [[]]
        for token in token_ids:
            token = int(token)
            if token == self.component_sep_token_id:
                components.append([])
            else:
                components[-1].append(token)
        return [] if components == [[]] else components

    @staticmethod
    def _record_node_label(labels_by_node: Dict[int, int], node: int, label: int) -> None:
        previous = labels_by_node.setdefault(node, label)
        if previous != label:
            raise ValueError(
                f"Feuler token stream assigns conflicting labels to node {node}.")

    def _read_graph(self, graph: Any) -> Dict[str, Any]:
        self._validate_graph_kind(graph)
        edge_index = self._get_graph_attr(graph, "edge_index")
        if edge_index is None:
            raise ValueError("Graph must provide edge_index.")

        edge_pairs = self._edge_pairs(edge_index)
        num_nodes = self._infer_num_nodes(graph, edge_pairs)
        for src, dst in edge_pairs:
            if src < 0 or dst < 0 or src >= num_nodes or dst >= num_nodes:
                raise ValueError(
                    "edge_index contains an endpoint outside the declared graph nodes.")
        node_labels = self._node_labels(self._get_graph_attr(graph, "x"), num_nodes)
        edge_labels = self._edge_labels(self._get_graph_attr(graph, "edge_attr"), len(edge_pairs))
        if self.protocol == "official" and any(label < 0 for label in node_labels + edge_labels):
            raise ValueError("Official node and edge token IDs must be non-negative.")
        input_edges = [(int(src), int(dst), int(edge_labels[i])) for i, (src, dst) in enumerate(edge_pairs)]
        canonical_edges = self._canonicalize_undirected_edges(input_edges)
        arcs = self._undirected_arcs(canonical_edges)
        return {
            "num_nodes": num_nodes,
            "node_labels": node_labels,
            "canonical_edges": canonical_edges,
            "input_edges": input_edges,
            "arcs": arcs,
            "node_signatures": (self._node_structure_signatures(
                num_nodes, node_labels, arcs) if self.protocol == "local_refs" else {}),
        }

    def _validate_graph_kind(self, graph: Any) -> None:
        validate_undirected_graph_semantics(graph)

    @staticmethod
    def _canonicalize_undirected_edges(
        input_edges: Sequence[Tuple[int, int, int]],
    ) -> List[Tuple[int, int, int]]:
        """Validate COO input and normalize it to one oriented edge per pair."""
        labels_by_edge: Dict[Tuple[int, int], int] = {}
        directed_edges = set()
        for src, dst, label in input_edges:
            if src == dst:
                raise ValueError("Feuler does not support self-loops.")
            directed_key = (src, dst)
            if directed_key in directed_edges:
                raise ValueError(
                    "Parallel edges are unsupported by the undirected simple "
                    "Feuler serializer.")
            directed_edges.add(directed_key)
            key = (min(src, dst), max(src, dst))
            previous = labels_by_edge.setdefault(key, label)
            if previous != label:
                raise ValueError(
                    "Feuler only supports undirected simple graphs; "
                    f"edge {key} has conflicting labels {previous} and {label}.")
        return [(src, dst, label) for (src, dst), label in sorted(labels_by_edge.items())]

    @staticmethod
    def _get_graph_attr(graph: Any, name: str):
        if isinstance(graph, dict):
            return graph.get(name)
        return getattr(graph, name, None)

    def _infer_num_nodes(self, graph: Any, edge_pairs: Sequence[Tuple[int, int]]) -> int:
        value = self._get_graph_attr(graph, "num_nodes")
        if callable(value):
            value = value()
        if value is not None:
            return int(value)
        x = self._get_graph_attr(graph, "x")
        if x is not None:
            return len(self._to_list(x))
        if not edge_pairs:
            return 0
        return max(max(src, dst) for src, dst in edge_pairs) + 1

    def _edge_pairs(self, edge_index: Any) -> List[Tuple[int, int]]:
        values = self._to_list(edge_index)
        if len(values) == 2 and self._is_sequence(values[0]) and self._is_sequence(values[1]):
            if len(values[0]) != len(values[1]):
                raise ValueError("edge_index source and destination rows must have the same length.")
            return [(int(src), int(dst)) for src, dst in zip(values[0], values[1])]
        return [(int(src), int(dst)) for src, dst in values]

    def _node_labels(self, x: Any, num_nodes: int) -> List[int]:
        if x is None:
            return [0] * num_nodes
        values = self._to_list(x)
        if len(values) != num_nodes:
            raise ValueError(
                f"x must contain exactly {num_nodes} node features, got {len(values)}.")
        return [self._scalar(value) for value in values]

    def _edge_labels(self, edge_attr: Any, num_edges: int) -> List[int]:
        if edge_attr is None:
            return [0] * num_edges
        values = self._to_list(edge_attr)
        if len(values) != num_edges:
            raise ValueError(
                f"edge_attr must contain exactly {num_edges} labels matching edge_index, got {len(values)}.")
        return [self._scalar(value) for value in values]

    def _undirected_arcs(self, input_edges: Sequence[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        arcs: List[Tuple[int, int, int]] = []
        for src, dst, label in input_edges:
            arcs.append((src, dst, label))
            arcs.append((dst, src, label))
        return arcs

    def _connected_components(self, num_nodes: int, arcs: Sequence[Tuple[int, int, int]]) -> List[List[int]]:
        neighbors: Dict[int, List[int]] = defaultdict(list)
        for src, dst, _ in arcs:
            neighbors[src].append(dst)
            neighbors[dst].append(src)

        components: List[List[int]] = []
        visited = set()
        for node in range(num_nodes):
            if node in visited:
                continue
            queue = deque([node])
            visited.add(node)
            component = []
            while queue:
                current = queue.popleft()
                component.append(current)
                for nxt in neighbors[current]:
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            components.append(component)
        return components

    def _serialize_component(
            self, graph_view: Dict[str, Any], component: Sequence[int],
            start_node: Optional[int] = None) -> List[Tuple[int, int, int]]:
        component_nodes = set(component)
        arcs = [
            (src, dst, label)
            for src, dst, label in graph_view["arcs"]
            if src in component_nodes and dst in component_nodes
        ]
        if not arcs:
            return []

        adjacency: Dict[int, List[Tuple[int, int, int, int]]] = defaultdict(list)
        for arc_id, (src, dst, label) in enumerate(arcs):
            adjacency[src].append((dst, label, arc_id, self._arc_priority(graph_view, src, dst, label)))
        for src in adjacency:
            adjacency[src].sort(key=lambda item: (
                -item[3], item[1], graph_view["node_signatures"][item[0]]))
            adjacency[src].reverse()

        start = (
            int(start_node) if start_node is not None
            else self._select_structural_node(
                graph_view, (node for node in component if adjacency.get(node))))
        if not adjacency.get(start):
            raise ValueError("start_node has no edge in this component.")
        used = set()
        circuit: List[Tuple[int, int, int]] = []
        node_stack = [start]
        edge_stack: List[Tuple[int, int, int]] = []
        while node_stack:
            node = node_stack[-1]
            while adjacency[node] and adjacency[node][-1][2] in used:
                adjacency[node].pop()
            if adjacency[node]:
                dst, label, arc_id, _ = adjacency[node].pop()
                used.add(arc_id)
                node_stack.append(dst)
                edge_stack.append((node, dst, label))
                continue
            node_stack.pop()
            if edge_stack:
                circuit.append(edge_stack.pop())

        circuit.reverse()
        return circuit

    def _arc_priority(self, graph_view: Dict[str, Any], src: int, dst: int, edge_label: int) -> int:
        pattern = self._edge_pattern(graph_view, src, dst, edge_label)
        return int(self.frequency_map.get(pattern, 0))

    @staticmethod
    def _edge_pattern(
        graph_view: Dict[str, Any], src: int, dst: int, edge_label: int,
    ) -> Tuple[int, int, int]:
        """Undirected frequency key; endpoint labels, never endpoint IDs, orient it."""
        src_label = graph_view["node_labels"][src]
        dst_label = graph_view["node_labels"][dst]
        return min(src_label, dst_label), int(edge_label), max(src_label, dst_label)

    @staticmethod
    def _node_structure_signatures(
        num_nodes: int,
        node_labels: Sequence[int],
        arcs: Sequence[Tuple[int, int, int]],
    ) -> Dict[int, Any]:
        """Refine node signatures using labels and labeled undirected neighborhoods.

        The signature is intentionally independent of the input node numbering.
        A tie after refinement does not prove that nodes are interchangeable.
        """
        neighbors: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for src, dst, edge_label in arcs:
            neighbors[src].append((int(edge_label), dst))
        signatures: Dict[int, int] = {
            node: int(node_labels[node]) for node in range(num_nodes)
        }
        for _ in range(num_nodes):
            descriptions = {
                node: (int(node_labels[node]), tuple(sorted(
                    (edge_label, signatures[neighbor])
                    for edge_label, neighbor in neighbors[node])))
                for node in range(num_nodes)
            }
            color_by_description = {
                description: color
                for color, description in enumerate(sorted(set(descriptions.values())))
            }
            refined = {
                node: color_by_description[description]
                for node, description in descriptions.items()
            }
            if refined == signatures:
                break
            signatures = refined
        return signatures

    @staticmethod
    def _select_structural_node(
        graph_view: Dict[str, Any], candidates: Iterable[int]) -> int:
        """Select by structure, retaining an arbitrary member only when tied."""
        selected = None
        selected_signature = None
        for node in candidates:
            signature = graph_view["node_signatures"][node]
            if selected is None or signature < selected_signature:
                selected = node
                selected_signature = signature
        if selected is None:
            raise ValueError("Feuler component has no node eligible for traversal.")
        return selected

    def _traversal_signature(
        self,
        graph_view: Dict[str, Any],
        edges: Sequence[Tuple[int, int, int]],
        component: Sequence[int],
    ) -> Tuple[Any, ...]:
        """Compare components without reversible raw node-reference tokens."""
        if not edges:
            node = self._select_structural_node(graph_view, component)
            return (("node", graph_view["node_labels"][node]),)
        signature: List[Any] = [("node", graph_view["node_labels"][edges[0][0]])]
        signature.extend(
            ("edge", edge_label, graph_view["node_labels"][dst])
            for _, dst, edge_label in edges)
        return tuple(signature)

    def _edges_to_tokens(
        self,
        graph_view: Dict[str, Any],
        edges: Sequence[Tuple[int, int, int]],
        component: Sequence[int],
        local_references: Dict[int, int],
        start_node: Optional[int] = None,
    ) -> List[int]:
        if not edges:
            return self._node_tokens(
                graph_view,
                int(start_node) if start_node is not None
                else self._select_structural_node(graph_view, component),
                local_references)

        tokens = self._node_tokens(graph_view, edges[0][0], local_references)
        for _, dst, edge_label in edges:
            if self.include_edge_tokens:
                tokens.append(self.edge_token(edge_label))
            tokens.extend(self._node_tokens(graph_view, dst, local_references))
        return tokens

    def _node_tokens(self, graph_view: Dict[str, Any], node: int,
                     local_references: Dict[int, int]) -> List[int]:
        if node not in local_references:
            local_references[node] = len(local_references)
        return [
            self.node_token(graph_view["node_labels"][node]),
            self.node_reference_token(local_references[node]),
        ]

    @staticmethod
    def _to_list(value: Any):
        if tlx.is_tensor(value):
            return tlx.convert_to_numpy(value).tolist()
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    @staticmethod
    def _is_sequence(value: Any) -> bool:
        return isinstance(value, Iterable) and not isinstance(value, (str, bytes))

    def _scalar(self, value: Any) -> int:
        value = self._to_list(value)
        if self._is_sequence(value):
            if len(value) != 1:
                raise ValueError(
                    "Multi-dimensional node and edge features require explicit "
                    "dataset-specific token encoding.")
            value = value[0]
            if self._is_sequence(value):
                raise ValueError(
                    "Multi-dimensional node and edge features require explicit "
                    "dataset-specific token encoding.")
        return self._strict_integer(
            value, "Node and edge labels must be integers.")

    @staticmethod
    def _strict_integer(value: Any, error_message: str) -> int:
        if isinstance(value, bool):
            raise ValueError(error_message)
        try:
            return operator.index(value)
        except TypeError as error:
            raise ValueError(error_message) from error


class EulerianSerializer(FrequencyGuidedEulerianSerializer):
    """Eulerian traversal without corpus frequency guidance (same protocols)."""

    def __init__(self, include_edge_tokens: bool = True, component_sep_token_id: Optional[int] = None,
                 protocol: str = "official", dataset: Optional[str] = None):
        super().__init__(
            name="eulerian",
            include_edge_tokens=include_edge_tokens,
            component_sep_token_id=component_sep_token_id,
            protocol=protocol, dataset=dataset,
        )

    def fit(self, graphs: Sequence[Any]):
        self.frequency_map = {}
        return self

    def _serialize_component(
        self,
        graph_view: Dict[str, Any],
        component: Sequence[int],
        start_node: Optional[int] = None,
    ) -> List[Tuple[int, int, int]]:
        component_nodes = set(component)
        arcs = [
            (src, dst, label)
            for src, dst, label in graph_view["arcs"]
            if src in component_nodes and dst in component_nodes
        ]
        if not arcs:
            return []

        adjacency: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for src, dst, label in arcs:
            adjacency[src].append((dst, label))
        for src in adjacency:
            adjacency[src].sort(
                key=lambda item: (item[1], graph_view["node_signatures"][item[0]]),
                reverse=True,
            )

        start = (
            int(start_node) if start_node is not None
            else self._select_structural_node(
                graph_view, (node for node in component if adjacency.get(node))))
        if not adjacency.get(start):
            raise ValueError("start_node has no edge in this component.")
        node_stack = [start]
        edge_stack: List[Tuple[int, int, int]] = []
        circuit: List[Tuple[int, int, int]] = []
        while node_stack:
            current = node_stack[-1]
            if adjacency[current]:
                dst, label = adjacency[current].pop()
                node_stack.append(dst)
                edge_stack.append((current, dst, label))
            else:
                node_stack.pop()
                if edge_stack:
                    circuit.append(edge_stack.pop())
        circuit.reverse()
        return circuit


GraphSerializer = FrequencyGuidedEulerianSerializer
