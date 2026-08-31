from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple


@dataclass
class GraphSerializationResult:
    """Container for serialized graph tokens and lightweight metadata."""

    token_ids: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)


class FrequencyGuidedEulerianSerializer:
    """Reversible Feuler serializer for undirected simple graphs only.

    ``edge_index`` may contain one or both directions of an undirected edge,
    but every direction must carry the same edge label. Directed graphs,
    parallel edges, and self-loops are not representable by this serializer.

    The token grammar is self-describing: a component starts with a node-label
    token and a node-reference token, then contains repeated
    ``edge-label, node-label, node-reference`` triples. Components are joined
    by ``component_sep_token_id``. Node labels, edge labels, and node
    references occupy disjoint integer domains, so the original graph can be
    recovered from the token stream alone.
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
            for src, dst, edge_label in graph_view["canonical_edges"]:
                counts[self._edge_pattern(graph_view, src, dst, edge_label)] += 1
        self.frequency_map = dict(counts)
        return self

    def serialize(self, graph: Any) -> GraphSerializationResult:
        graph_view = self._read_graph(graph)
        components = self._connected_components(graph_view["num_nodes"], graph_view["arcs"])
        component_results = []

        for component in components:
            edges = self._serialize_component(graph_view, component)
            tokens = self._edges_to_tokens(graph_view, edges, component)
            component_results.append((
                tokens,
                len(edges),
                self._traversal_signature(graph_view, edges, component),
            ))

        component_results.sort(key=lambda item: (-len(item[0]), item[2]))
        token_ids: List[int] = []
        for index, (component_tokens, _, _) in enumerate(component_results):
            if index > 0:
                token_ids.append(self.component_sep_token_id)
            token_ids.extend(component_tokens)

        return GraphSerializationResult(
            token_ids=token_ids,
            metadata={
                "method": self.name,
                "protocol_version": 2,
                "num_nodes": graph_view["num_nodes"],
                "num_components": len(component_results),
                "num_edges_traversed": sum(
                    edge_count for _, edge_count, _ in component_results),
            },
        )

    def restore_input_metadata(self, result: GraphSerializationResult) -> Dict[str, List[int]]:
        """Backward-compatible alias for token-stream reconstruction."""
        return self.deserialize(result)

    def deserialize(self, result: GraphSerializationResult) -> Dict[str, List[int]]:
        """Reconstruct a graph from a Feuler token stream without metadata."""
        if not self.include_edge_tokens:
            raise ValueError("Feuler deserialization requires include_edge_tokens=True.")
        token_ids = result.token_ids if isinstance(result, GraphSerializationResult) else result
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
        input_edges = [(int(src), int(dst), int(edge_labels[i])) for i, (src, dst) in enumerate(edge_pairs)]
        canonical_edges = self._canonicalize_undirected_edges(input_edges)
        arcs = self._undirected_arcs(canonical_edges)
        return {
            "num_nodes": num_nodes,
            "node_labels": node_labels,
            "canonical_edges": canonical_edges,
            "arcs": arcs,
            "node_signatures": self._node_structure_signatures(
                num_nodes, node_labels, arcs),
        }

    def _validate_graph_kind(self, graph: Any) -> None:
        """Reject graph objects that explicitly declare directed semantics.

        A single COO orientation remains a supported *storage* form for one
        undirected edge.  It is indistinguishable from a directed edge without
        graph metadata, so objects that expose ``directed`` or ``is_directed``
        must declare it false.
        """
        for name in ("directed", "is_directed"):
            value = self._get_graph_attr(graph, name)
            if callable(value):
                value = value()
            if value is not None and bool(value):
                raise ValueError(
                    "Feuler only supports undirected simple graphs; "
                    "directed graph semantics are unsupported.")

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
            return list(range(num_nodes))
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
            adjacency[src].sort(key=lambda item: (
                -item[3], item[1], graph_view["node_signatures"][item[0]]))
            adjacency[src].reverse()

        start = self._select_structural_node(
            graph_view, (node for node in component if adjacency.get(node)))
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
        Nodes that remain tied after refinement are structurally equivalent for
        the supported labeled simple-graph contract.
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
    ) -> List[int]:
        if not edges:
            return self._node_tokens(
                graph_view, self._select_structural_node(graph_view, component))

        tokens = self._node_tokens(graph_view, edges[0][0])
        for _, dst, edge_label in edges:
            if self.include_edge_tokens:
                tokens.append(self.edge_token(edge_label))
            tokens.extend(self._node_tokens(graph_view, dst))
        return tokens

    def _node_tokens(self, graph_view: Dict[str, Any], node: int) -> List[int]:
        return [
            self.node_token(graph_view["node_labels"][node]),
            self.node_reference_token(node),
        ]

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
            adjacency[src].sort(
                key=lambda item: (item[1], graph_view["node_signatures"][item[0]]),
                reverse=True,
            )

        start = self._select_structural_node(
            graph_view, (node for node in component if adjacency.get(node)))
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
