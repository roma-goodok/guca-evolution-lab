# file: src/gum_graph.py

import networkx as nx

class GUMGraph:
    """
    A higher-level wrapper around a networkx Graph,
    for GUCA-specific functionality (states, births, connections, etc.).
    """

    def __init__(self, max_vertices=0, max_connections=16):
        """
        :param max_vertices: Optional limit on total nodes. 
                             0 means 'no limit'.
        :param max_connections: A cap on the degree any node should reach.
        """
        self._graph = nx.Graph()
        self.max_vertices = max_vertices
        self.max_connections = max_connections

    def add_node(self, node_id, state, prior_state=None, parents_count=0, **kwargs):
        """
        Add a new GUM node with specific state/tracking info.
        
        :param node_id: A unique identifier or integer index.
        :param state: Current state (e.g. NodeState.A).
        :param prior_state: The node's previous state.
        :param parents_count: How many 'generations' or births lead to this node.
        :param kwargs: Any additional attributes you may want to store.
        """
        if self.max_vertices and len(self._graph) >= self.max_vertices:
            # Reached node limit
            return None
        
        self._graph.add_node(node_id,
                             state=state,
                             priorState=prior_state,
                             parentsCount=parents_count,
                             markedAsDeleted=False,
                             **kwargs)
        return node_id

    def remove_node(self, node_id):
        """
        Mark node as deleted or remove it from the graph altogether.
        """
        if node_id in self._graph:
            self._graph.remove_node(node_id)

    def add_edge(self, source, target):
        """
        Add an edge, respecting the max_connections limit if needed.
        """
        if (source not in self._graph) or (target not in self._graph):
            return  # invalid node(s)

        # Check node degrees before connecting
        if (self.max_connections > 0):
            if self._graph.degree[source] >= self.max_connections:
                return
            if self._graph.degree[target] >= self.max_connections:
                return

        self._graph.add_edge(source, target)

    def remove_edge(self, source, target):
        """
        Remove an existing edge, if present.
        """
        if self._graph.has_edge(source, target):
            self._graph.remove_edge(source, target)

    def get_state(self, node_id):
        """
        Helper to retrieve the node's current state.
        """
        return self._graph.nodes[node_id].get('state', None)

    def set_state(self, node_id, new_state):
        """
        Update the node's current state.
        """
        if node_id in self._graph:
            self._graph.nodes[node_id]['state'] = new_state

    def get_prior_state(self, node_id):
        """
        Retrieve the node's priorState attribute.
        """
        return self._graph.nodes[node_id].get('priorState', None)

    def set_prior_state(self, node_id, prior_state):
        """
        Update the node's priorState attribute.
        """
        if node_id in self._graph:
            self._graph.nodes[node_id]['priorState'] = prior_state

    def is_deleted(self, node_id):
        """
        Check if the node is marked as deleted.
        """
        return self._graph.nodes[node_id].get('markedAsDeleted', False)

    def mark_as_deleted(self, node_id):
        """
        Mark the node as deleted (you can later remove it physically if you wish).
        """
        if node_id in self._graph:
            self._graph.nodes[node_id]['markedAsDeleted'] = True

    def node_degree(self, node_id):
        """
        Return how many edges are connected to this node.
        """
        return self._graph.degree[node_id]

    def node_parents_count(self, node_id):
        """
        Return the 'parentsCount' or generation-level for this node.
        """
        return self._graph.nodes[node_id].get('parentsCount', 0)

    def set_node_parents_count(self, node_id, value):
        if node_id in self._graph:
            self._graph.nodes[node_id]['parentsCount'] = value

    def get_neighbors(self, node_id):
        """
        Return a list of neighbors for the specified node.
        """
        if node_id in self._graph:
            return list(self._graph.neighbors(node_id))
        return []

    def get_underlying_graph(self):
        """
        Return the underlying networkx Graph object,
        if you need direct access to more networkx features.
        """
        return self._graph

    def __len__(self):
        """
        Number of nodes in the graph.
        """
        return len(self._graph)

    def __contains__(self, node_id):
        """
        Check if a node is in this GUMGraph.
        """
        return node_id in self._graph
