use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use super::graph;

// Given a vec of Nodes in P2P network, create a directed graph
// Directions in the graph represent compatibility between layer ranges
//
// # Arguments
//
// * `nodes` - Vec of nodes in network
//
// # Returns
//
//  Verticies and edges of graph
pub fn make_graph(nodes: &Vec<P2PNode>) -> graph::DirectedGraph<&P2PNode> {
    let mut graph = graph::DirectedGraph::new();

    // Add all nodes to vertices
    for node in nodes {
        graph.add_node(node);
    }

    // Create edges based on layer range compatibility
    for source_node in nodes {
        for target_node in nodes {
            // Skip self loops
            if source_node.id == target_node.id {
                continue;
            }

            // Create an edge if target node's layer range is one greater than source node's
            if target_node.params.layer_range == source_node.params.layer_range + 1 {
                graph.add_edge(source_node, target_node);
            }
        }
    }

    graph
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct P2PNode {
    pub id: usize,
    pub params: NodeParameters,
}

impl P2PNode {
    /// Creates a new P2PNode with the given id and parameters.
    ///
    /// # Arguments
    ///
    /// * `id` - A unique identifier for this node
    /// * `params` - The parameters that configure this node's behavior
    ///
    /// # Returns
    ///
    /// A new P2PNode instance
    pub fn new(id: usize, params: NodeParameters) -> Self {
        P2PNode { id, params }
    }
}

/// Implements the Hash trait for P2PNode.
///
/// This implementation only hashes the `id` field, ignoring the `params` field.
/// This means two P2PNodes with the same id will hash to the same value,
/// regardless of their parameters.
impl Hash for P2PNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

// Represents the parameters for a node in a distributed system.
///
/// # Fields
///
/// * `layer_range` - Range of layers node i has
/// * `latencies` - Map of latencies from this node to other nodes, where the key is the node ID
/// * `computational_cost` - Computational cost of the node
/// * `preload_cost` - Preload cost of the node
/// * `embedding_cost` - Embedding cost of the node
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct NodeParameters {
    /// Range of layers node serves (l_i)
    pub layer_range: u8,

    /// Latencies from this node to other nodes, where the key is the destination node ID
    /// ℓᵢⱼ represents the latency from node i to node j
    pub latencies: HashMap<usize, u32>,

    /// Computational cost of the node (cᵢ)
    pub computational_cost: u32,

    /// Preload cost of the node (pᵢ)
    pub preload_cost: u32,

    /// Embedding cost of the node (eᵢ)
    pub embedding_cost: u32,
}

impl NodeParameters {
    /// Creates a new NodeParameters instance
    ///
    /// # Arguments
    ///
    /// * `layer_start` - The starting layer index (must be less than `layer_end`)
    /// * `layer_end` - The ending layer index (must be less than L and greater than `layer_start`)
    /// * `computational_cost` - Computational cost value
    /// * `preload_cost` - Preload cost value
    /// * `embedding_cost` - Embedding cost value
    ///
    /// # Returns
    ///
    /// A new NodeParameters instance with empty latencies map
    pub fn new(
        layer_range: u8,
        computational_cost: u32,
        preload_cost: u32,
        embedding_cost: u32,
    ) -> Result<Self, &'static str> {
        Ok(NodeParameters {
            layer_range,
            latencies: HashMap::new(),
            computational_cost,
            preload_cost,
            embedding_cost,
        })
    }

    /// Sets the latency from this node to another node
    ///
    /// # Arguments
    ///
    /// * `destination_node_id` - The ID of the destination node
    /// * `latency` - The latency value
    pub fn set_latency(&mut self, destination_node_id: usize, latency: u32) {
        self.latencies.insert(destination_node_id, latency);
    }

    /// Gets the latency from this node to another node
    ///
    /// # Arguments
    ///
    /// * `destination_node_id` - The ID of the destination node
    ///
    /// # Returns
    ///
    /// The latency value if it exists, None otherwise
    pub fn get_latency(&self, destination_node_id: &usize) -> Option<&u32> {
        self.latencies.get(destination_node_id)
    }
}
