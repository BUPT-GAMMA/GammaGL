from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple


@dataclass
class GraphSerializationResult:
    """Container for serialized graph tokens and lightweight metadata."""

    token_ids: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)


class FrequencyGuidedEulerianSerializer:
    """Paper Feuler serializer for undirected simple graphs only.

    ``edge_index`` may contain one or both directions of an undirected edge,
    but every direction must carry the same edge label. Directed graphs,
    parallel edges, and self-loops are not representable by this serializer.
    """

    def __init__(
        self,
        name: str = "feuler",
        include_edge_tokens: bool = True,
        component_sep_token_id: int = -1,
    ):
        self.name = name
        self.include_edge_tokens = include_edge_tokens
        self.component_sep_token_id = int(component_sep_token_id)
        self.frequency_map: Dict[Any, int] = {}

    def fit(self, graphs: Sequence[Any]):
        counts: Dict[Tuple[int, int, int], int] = defaultdict(int)
        for graph in graphs:
            graph_view = self._read_graph(graph)
            for src, dst, edge_label in graph_view["input_edges"]:
                counts[(graph_view["node_labels"][src], edge_label, graph_view["node_labels"][dst])] += 1
        self.frequency_map = dict(counts)
        return self

    def serialize(self, graph: Any) -> GraphSerializationResult:
        graph_view = self._read_graph(graph)
        components = self._connected_components(graph_view["num_nodes"], graph_view["arcs"])
        component_results = []

        for component in components:
            edges = self._serialize_component(graph_view, component)
            tokens = self._edges_to_tokens(graph_view, edges, component)
            component_results.append((tokens, len(edges)))

        component_results.sort(key=lambda item: (-len(item[0]), item[0]))
        token_ids: List[int] = []
        for index, (component_tokens, _) in enumerate(component_results):
            if index > 0:
                token_ids.append(self.component_sep_token_id)
            token_ids.extend(component_tokens)

        return GraphSerializationResult(
            token_ids=token_ids,
            metadata={
                "method": self.name,
                "num_nodes": graph_view["num_nodes"],
                "num_components": len(component_results),
                "num_edges_traversed": sum(edge_count for _, edge_count in component_results),
                "node_labels": list(graph_view["node_labels"]),
                "input_edges": [list(edge) for edge in graph_view["input_edges"]],
            },
        )

    def restore_input_metadata(self, result: GraphSerializationResult) -> Dict[str, List[int]]:
        """Return input metadata; this is not a token-stream decoder."""
        metadata = result.metadata
        if "node_labels" not in metadata or "input_edges" not in metadata:
            raise ValueError("Serialization result does not contain round-trip metadata.")
        input_edges = [tuple(int(value) for value in edge) for edge in metadata["input_edges"]]
        return {
            "edge_index": [
                [src for src, _, _ in input_edges],
                [dst for _, dst, _ in input_edges],
            ],
            "x": [int(label) for label in metadata["node_labels"]],
            "edge_attr": [label for _, _, label in input_edges],
        }

    def deserialize(self, result: GraphSerializationResult) -> Dict[str, List[int]]:
        """Reject ambiguous use as a token-stream decoder."""
        raise NotImplementedError(
            "Feuler token streams are not self-describing; use "
            "restore_input_metadata() for metadata validation.")

    @staticmethod
    def node_token(label: int) -> int:
        return int(label)

    @staticmethod
    def edge_token(label: int) -> int:
        return int(label)

    def _read_graph(self, graph: Any) -> Dict[str, Any]:
        edge_index = self._get_graph_attr(graph, "edge_index")
        if edge_index is None:
            raise ValueError("Graph must provide edge_index.")

        edge_pairs = self._edge_pairs(edge_index)
        num_nodes = self._infer_num_nodes(graph, edge_pairs)
        node_labels = self._node_labels(self._get_graph_attr(graph, "x"), num_nodes)
        edge_labels = self._edge_labels(self._get_graph_attr(graph, "edge_attr"), len(edge_pairs))
        input_edges = [(int(src), int(dst), int(edge_labels[i])) for i, (src, dst) in enumerate(edge_pairs)]
        self._validate_undirected_simple(input_edges)
        arcs = self._undirected_arcs(input_edges)
        return {
            "num_nodes": num_nodes,
            "node_labels": node_labels,
            "input_edges": input_edges,
            "arcs": arcs,
        }

    @staticmethod
    def _validate_undirected_simple(input_edges: Sequence[Tuple[int, int, int]]) -> None:
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
            return [(int(src), int(dst)) for src, dst in zip(values[0], values[1])]
        return [(int(src), int(dst)) for src, dst in values]

    def _node_labels(self, x: Any, num_nodes: int) -> List[int]:
        if x is None:
            return list(range(num_nodes))
        values = self._to_list(x)
        return [self._scalar(values[i]) if i < len(values) else i for i in range(num_nodes)]

    def _edge_labels(self, edge_attr: Any, num_edges: int) -> List[int]:
        if edge_attr is None:
            return [0] * num_edges
        values = self._to_list(edge_attr)
        return [self._scalar(values[i]) if i < len(values) else 0 for i in range(num_edges)]

    def _undirected_arcs(self, input_edges: Sequence[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        seen = set()
        arcs: List[Tuple[int, int, int]] = []
        for src, dst, label in input_edges:
            if src == dst:
                key = (src, dst, label)
                if key not in seen:
                    seen.add(key)
                    arcs.append((src, dst, label))
                continue
            key = (min(src, dst), max(src, dst), label)
            if key in seen:
                continue
            seen.add(key)
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
            components.append(sorted(component))
        return components

    def _serialize_component(self, graph_view: Dict[str, Any], component: Sequence[int]) -> List[Tuple[int, int, int]]:
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
            adjacency[src].sort(key=lambda item: (-item[3], item[0]))
            adjacency[src].reverse()

        start = min(node for node in component if adjacency.get(node))
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
        pattern = (graph_view["node_labels"][src], edge_label, graph_view["node_labels"][dst])
        return int(self.frequency_map.get(pattern, 0))

    def _edges_to_tokens(
        self,
        graph_view: Dict[str, Any],
        edges: Sequence[Tuple[int, int, int]],
        component: Sequence[int],
    ) -> List[int]:
        if not edges:
            return [self.node_token(graph_view["node_labels"][component[0]])]

        tokens = [self.node_token(graph_view["node_labels"][edges[0][0]])]
        for _, dst, edge_label in edges:
            if self.include_edge_tokens:
                tokens.append(self.edge_token(edge_label))
            tokens.append(self.node_token(graph_view["node_labels"][dst]))
        return tokens

    @staticmethod
    def _to_list(value: Any):
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
        return int(value)


class EulerianSerializer(FrequencyGuidedEulerianSerializer):
    """Deterministic Eulerian serializer without corpus frequency guidance."""

    def __init__(self, include_edge_tokens: bool = True, component_sep_token_id: int = -1):
        super().__init__(
            name="eulerian",
            include_edge_tokens=include_edge_tokens,
            component_sep_token_id=component_sep_token_id,
        )

    def fit(self, graphs: Sequence[Any]):
        self.frequency_map = {}
        return self

    def _serialize_component(
        self,
        graph_view: Dict[str, Any],
        component: Sequence[int],
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
            adjacency[src].sort(key=lambda item: (item[0], item[1]))

        start = min(node for node in component if adjacency.get(node))
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
