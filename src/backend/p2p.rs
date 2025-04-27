use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};

use super::graph;

/// Represents a peer-to-peer network composed of nodes with specific layer ranges
/// and connectivity information.
///
/// # Fields
///
/// * `nodes` - All nodes in the network
/// * `start_node_ids` - IDs of nodes that serve as entry points to the network (at min_layer)
/// * `end_node_ids` - IDs of nodes that serve as terminal points in the network (at max_layer)
/// * `min_layer` - The minimum layer range in the network
/// * `max_layer` - The maximum layer range in the network
pub struct P2PNetwork<'a> {
    pub nodes: Vec<P2PNode>,
    pub start_node_ids: Vec<usize>,
    pub end_node_ids: Vec<usize>,
    pub min_layer: u8,
    pub max_layer: u8,
    pub contracts: Vec<Contract<'a>>,
}

impl<'a> P2PNetwork<'a> {
    /// Creates a new P2PNetwork with empty node vectors and default layer bounds.
    ///
    /// # Returns
    ///
    /// A new P2PNetwork instance with empty node collections and u8::MAX/u8::MIN as initial bounds
    pub fn new() -> Self {
        P2PNetwork {
            nodes: Vec::new(),
            start_node_ids: Vec::new(),
            end_node_ids: Vec::new(),
            min_layer: u8::MAX,
            max_layer: u8::MIN,
            contracts: Vec::new(),
        }
    }

    /// Creates a new P2PNetwork from a list of nodes, automatically determining
    /// which nodes should be considered start nodes and end nodes based on their layer ranges.
    ///
    /// # Arguments
    ///
    /// * `nodes` - Nodes to include in the network
    ///
    /// # Returns
    ///
    /// A new P2PNetwork with nodes categorized appropriately
    pub fn from_nodes(nodes: Vec<P2PNode>) -> Self {
        if nodes.is_empty() {
            return Self::new();
        }

        // Find the minimum and maximum layer ranges
        let min_layer = nodes.iter().map(|n| n.params.layer_range).min().unwrap();
        let max_layer = nodes.iter().map(|n| n.params.layer_range).max().unwrap();

        let start_node_ids = nodes
            .iter()
            .filter(|n| n.params.layer_range == min_layer)
            .map(|n| n.id)
            .collect();

        let end_node_ids = nodes
            .iter()
            .filter(|n| n.params.layer_range == max_layer)
            .map(|n| n.id)
            .collect();

        P2PNetwork {
            nodes,
            start_node_ids,
            end_node_ids,
            min_layer,
            max_layer,
            contracts: Vec::new(),
        }
    }

    /// Adds a node to the network, automatically updating start_node_ids, end_node_ids,
    /// min_layer, and max_layer if needed.
    ///
    /// # Arguments
    ///
    /// * `node` - The node to add to the network
    pub fn add_node(&mut self, node: P2PNode) {
        let node_id = node.id;
        let node_layer = node.params.layer_range;

        // Update min_layer and start_node_ids if needed
        if self.nodes.is_empty() || node_layer < self.min_layer {
            self.min_layer = node_layer;
            self.start_node_ids.clear();
            self.start_node_ids.push(node_id);
        } else if node_layer == self.min_layer {
            self.start_node_ids.push(node_id);
        }

        // Update max_layer and end_node_ids if needed
        if self.nodes.is_empty() || node_layer > self.max_layer {
            self.max_layer = node_layer;
            self.end_node_ids.clear();
            self.end_node_ids.push(node_id);
        } else if node_layer == self.max_layer {
            self.end_node_ids.push(node_id);
        }

        // Add the node to the main nodes collection
        self.nodes.push(node);
    }

    // Changes a nodes layer range in the network, automatically updating start_node_ids,
    // end_node_ids if needed
    //
    // # Arguments
    //
    // * `node` - The node to change in the network
    // * `new_layer` - The new layer range to set
    pub fn change_node_layer(&mut self, node: &mut P2PNode, new_layer: u8) {
        let node_id = node.id;
        let current_layer = node.params.layer_range;
        node.params.layer_range = new_layer;

        // Remove node if previously was start or end
        if current_layer == self.min_layer {
            self.start_node_ids.retain(|id| *id != node_id);
        } else if current_layer == self.max_layer {
            self.end_node_ids.retain(|id| *id != node_id);
        }

        // Add to aprropriate list if new range is start or end
        if new_layer == self.min_layer {
            self.start_node_ids.push(node_id);
        } else if new_layer == self.max_layer {
            self.end_node_ids.push(node_id);
        }
    }

    /// Given the current nodes in the P2P network, create a directed graph.
    /// Directions in the graph represent compatibility between layer ranges,
    /// where an edge exists if the target node's layer range is one greater than
    /// the source node's layer range.
    ///
    /// # Returns
    ///
    /// DirectedGraph representing the network topology
    pub fn make_graph(&self) -> graph::DirectedGraph<&P2PNode> {
        let mut graph = graph::DirectedGraph::new();

        // Add all nodes to vertices
        for node in &self.nodes {
            graph.add_node(node);
        }

        // Create edges based on layer range compatibility
        for source_node in &self.nodes {
            for target_node in &self.nodes {
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

    /// Gets references to all start nodes (nodes with the minimum layer range).
    ///
    /// # Returns
    ///
    /// A vector of references to the start nodes
    pub fn start_nodes(&self) -> Vec<&P2PNode> {
        self.start_node_ids
            .iter()
            .filter_map(|id| self.find_node_by_id(*id))
            .collect()
    }

    /// Gets references to all end nodes (nodes with the maximum layer range).
    ///
    /// # Returns
    ///
    /// A vector of references to the end nodes
    pub fn end_nodes(&self) -> Vec<&P2PNode> {
        self.end_node_ids
            .iter()
            .filter_map(|id| self.find_node_by_id(*id))
            .collect()
    }

    /// Finds a node in the network by its ID.
    ///
    /// # Arguments
    ///
    /// * `id` - The ID of the node to find
    ///
    /// # Returns
    ///
    /// Option containing a reference to the node if found, None otherwise
    pub fn find_node_by_id(&self, id: usize) -> Option<&P2PNode> {
        self.nodes.iter().find(|node| node.id == id)
    }
}

/// Represents a sub-contract within a larger contract system
///
/// A `SubContract` defines a relationship between a source and destination entity
/// with a time constraint for completion.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SubContract {
    /// Identifier for the source entity
    pub source_id: usize,

    /// Owner of the contract doing the computation
    pub owner_id: usize,

    /// Identifier for the destination entity
    pub dest_id: usize,

    /// Time remaining (in seconds) before the sub-contract expires
    pub time_left: u32,
}

impl SubContract {
    /// Creates a new sub-contract
    ///
    /// # Arguments
    ///
    /// * `source_id` - The ID of the source entity
    /// * `dest_id` - The ID of the destination entity
    /// * `time_left` - Time until expiration in seconds
    ///
    /// # Returns
    ///
    /// A new `SubContract` instance
    pub fn new(source_id: usize, owner_id: usize, dest_id: usize, time_left: u32) -> Self {
        SubContract {
            source_id,
            owner_id,
            dest_id,
            time_left,
        }
    }

    /// Decrements the time left on the sub-contract
    ///
    /// # Arguments
    ///
    /// * `seconds` - The number of seconds to decrement
    ///
    /// # Returns
    ///
    /// `true` if time remains, `false` if expired
    pub fn tick(&mut self, seconds: u32) -> bool {
        if seconds >= self.time_left {
            self.time_left = 0;
            false
        } else {
            self.time_left -= seconds;
            true
        }
    }

    /// Checks if the sub-contract has completed
    ///
    /// # Returns
    ///
    /// `true` if expired, `false` otherwise
    pub fn is_complete(&self) -> bool {
        self.time_left == 0
    }
}

/// Represents a multi-layered contract on a blockchain
///
/// A `Contract` consists of multiple sub-contracts that must be fulfilled
/// in sequence. The contract tracks fulfillment status and manages the
/// relationship between all sub-contracts.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Contract<'a> {
    fulfilled: bool,
    layers: Vec<&'a SubContract>,
}

impl<'a> Contract<'a> {
    /// Creates a new contract with specified sub-contracts
    ///
    /// # Arguments
    ///
    /// * `layers` - A vector of references to `SubContract`s
    ///
    /// # Returns
    ///
    /// A new `Contract` instance
    pub fn new(layers: Vec<&'a SubContract>) -> Self {
        Contract {
            fulfilled: false,
            layers,
        }
    }

    /// Creates an empty contract with no sub-contracts
    ///
    /// # Returns
    ///
    /// A new empty `Contract` instance
    pub fn empty() -> Self {
        Contract {
            fulfilled: false,
            layers: Vec::new(),
        }
    }

    /// Checks if the contract is fulfilled
    ///
    /// # Returns
    ///
    /// `true` if the contract is fulfilled, `false` otherwise
    pub fn is_fulfilled(&self) -> bool {
        self.fulfilled
    }

    /// Attempts to fulfill the contract
    ///
    /// This method checks if all sub-contracts are valid and not expired,
    /// and if so, marks the contract as fulfilled.
    ///
    /// # Returns
    ///
    /// `true` if the contract was fulfilled, `false` otherwise
    pub fn fulfill(&mut self) -> bool {
        // Cannot fulfill an empty contract
        if self.layers.is_empty() {
            return false;
        }

        // Check if any sub-contracts are expired
        for layer in &self.layers {
            if layer.is_complete() {
                return false;
            }
        }

        // Mark contract as fulfilled
        self.fulfilled = true;
        true
    }

    /// Validates the contract's integrity
    ///
    /// Checks that each sub-contract's destination connects to the next
    /// sub-contract's source, forming a complete chain.
    ///
    /// # Returns
    ///
    /// `true` if the contract forms a valid chain, `false` otherwise
    pub fn validate(&self) -> bool {
        if self.layers.len() <= 1 {
            return true;
        }

        for i in 0..self.layers.len() - 1 {
            if self.layers[i].dest_id != self.layers[i + 1].source_id {
                return false;
            }
        }

        true
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct P2PNode {
    pub id: usize,
    pub params: NodeParameters,
    pub queue: VecDeque<SubContract>,
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
        P2PNode {
            id,
            params,
            queue: VecDeque::new(),
        }
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
    ) -> Self {
        NodeParameters {
            layer_range,
            latencies: HashMap::new(),
            computational_cost,
            preload_cost,
            embedding_cost,
        }
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
