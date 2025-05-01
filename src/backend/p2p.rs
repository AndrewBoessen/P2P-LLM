use super::graph::{self, DirectedGraph};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};

fn price_gradient(old: f64, prob: f64) -> f64 {
    prob + old * 0.5 * prob * (prob - 1.0)
}
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
/// * `contracts` - List of contracts the network has processed
pub struct P2PNetwork {
    pub nodes: Vec<P2PNode>,
    pub start_node_ids: Vec<usize>,
    pub end_node_ids: Vec<usize>,
    pub min_layer: u8,
    pub max_layer: u8,
    pub contracts: Vec<Contract>,
}

impl P2PNetwork {
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

    pub fn nodes_without_contracts(&self) -> Vec<&P2PNode> {
        // check if there is an active contract for every contract
        // otherwise return the nodes without active contracts
        let mut nodes = vec![true; self.nodes.len()];

        for contract in self.contracts.iter() {
            if contract.fulfilled == false {
                nodes[contract.owner] = false;
            }
        }

        nodes
            .iter()
            .enumerate()
            .filter_map(|(node_id, value)| {
                value.then_some(
                    P2PNetwork::find_node_by_id(self, node_id).expect("node not in network"),
                )
            })
            .collect()
    }

    /// Update the state of the network after given time period
    ///
    /// # Arguments
    ///
    /// * `period` - Time in miliseconds
    /// * `iter` - The current iteration in the simulation
    /// * `layer_period` - Iteration between finding optimal layer allocation
    pub fn update_network(
        &mut self,
        period: u32,
        iter: u32,
        layer_period: u32,
        node_to_update: usize,
    ) {
        // State of each nodes currently
        let mut states = vec![false; self.nodes.len()];

        // Process each contract
        let mut contracts_to_update = Vec::new();

        for contract in self.contracts.iter_mut() {
            if !contract.fulfilled {
                if let Some(sub) = contract.layers.front_mut() {
                    // owner is in use this round
                    if !states[sub.owner_id] {
                        states[sub.owner_id] = true;

                        if !SubContract::tick(sub, period) {
                            // Store the IDs for later balance update
                            contracts_to_update.push((sub.price, sub.owner_id, contract.owner));
                            // Remove the front layer
                            contract.layers.pop_front();
                        }
                    }
                }

                Contract::fulfill(contract);
            }
        }

        // Update balances using the collected information
        for (price, seller_id, buyer_id) in contracts_to_update {
            let seller = self
                .find_node_by_id_mut(seller_id)
                .expect("owner of contract not found");
            seller.balance += price;

            let buyer = self
                .find_node_by_id_mut(buyer_id)
                .expect("owner of sub contract not found");
            buyer.balance -= price;
        }

        // udpate prices that node charge
        let mut new_prices = vec![0.0; self.nodes.len()];
        let mut new_layers = vec![0u8; self.nodes.len()];
        for node in self.nodes.iter() {
            let new_price = P2PNetwork::update_price(self, node);
            new_prices[node.id] = new_price;

            if iter % layer_period == 0 {
                new_layers[node.id] = P2PNetwork::find_optimal_layer(self, node);
            }
        }

        for node_mut in self.nodes.iter_mut() {
            node_mut.price = new_prices[node_mut.id];

            if node_mut.id == node_to_update && iter % layer_period == 0 {
                // only update if different
                if new_layers[node_mut.id] != node_mut.params.layer_range {
                    P2PNetwork::change_node_layer(
                        &mut self.start_node_ids,
                        &mut self.end_node_ids,
                        self.min_layer,
                        self.max_layer,
                        node_mut,
                        new_layers[node_mut.id],
                    );
                }
            }
        }
    }

    /// Find share of subcontracts that a node will recieve
    /// based on the computational costs
    ///
    /// # Arguments
    ///
    /// * `node` - target node
    /// * `price` - price that node will charge
    ///
    /// # Returns
    ///
    /// The percentage of market share the node has
    pub fn market_share(&self, node: &P2PNode, price: f64) -> f64 {
        let layer = node.params.layer_range;
        let cost = node.params.computational_cost;

        let cost = 0.5 * price + 0.5 * (cost as f64 / 50.0);

        let nodes_in_layer: Vec<f64> = self
            .nodes
            .iter()
            .filter_map(|node: &P2PNode| {
                (node.params.layer_range == layer).then_some(
                    (-(0.5 * node.price + 0.5 * (node.params.computational_cost as f64 / 50.0)))
                        .exp(),
                )
            })
            .collect();

        let sum: f64 = nodes_in_layer.iter().sum();

        // softmax function
        (-cost).exp() as f64 / sum as f64
    }

    /// Finds the volume of computation in the layer
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer to get average on
    ///
    /// # Returns
    ///
    /// Sum of total computation cost in layer
    fn layer_volume(&self, layer: u8) -> f64 {
        let nodes_in_layer: Vec<f64> = self
            .nodes
            .iter()
            .filter_map(|node: &P2PNode| {
                (node.params.layer_range == layer)
                    .then_some((-(node.params.computational_cost as f64 / 50.0)).exp())
            })
            .collect();

        nodes_in_layer.iter().sum()
    }

    /// Find the layer with optimal balance between node count and computational cost
    ///
    /// # Arguments
    ///
    /// * `node` - node to optimize for
    ///
    /// # Returns
    ///
    /// Layer range that optimizes balance between node distribution and computational cost
    pub fn find_optimal_layer(&self, node: &P2PNode) -> u8 {
        let node_layer = node.params.layer_range;
        let node_comp = (-(node.params.computational_cost as f64 / 50.0)).exp();

        // Track the best layer and its score
        let mut best_layer = node_layer;
        let mut best_score = f64::MIN;

        // Find layer that balances both node distribution and computational cost similarity
        for layer in self.min_layer..=self.max_layer {
            let share: f64;
            let curr_vol = P2PNetwork::layer_volume(self, layer);
            if layer == node_layer {
                share = node_comp / (curr_vol - node_comp);
            } else {
                share = node_comp / curr_vol;
            }
            if share > best_score {
                best_score = share;
                best_layer = layer;
            }
        }

        // Return the layer with the best balance score
        best_layer
    }

    /// Updates price to maximize revenue
    /// uses Newton method for find maximum of epxected revenue
    ///
    /// # Arguments
    ///
    /// * 'node' - node id to update
    ///
    /// # Returns
    ///
    /// New price to use
    pub fn update_price(&self, node: &P2PNode) -> f64 {
        let cur_price = node.price;

        let prob = P2PNetwork::market_share(self, node, node.price);

        cur_price + price_gradient(cur_price, prob)
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
    fn change_node_layer(
        start_node_ids: &mut Vec<usize>,
        end_node_ids: &mut Vec<usize>,
        min_layer: u8,
        max_layer: u8,
        node: &mut P2PNode,
        new_layer: u8,
    ) {
        let node_id = node.id;
        let current_layer = node.params.layer_range;
        node.params.layer_range = new_layer;

        // Remove node if previously was start or end
        if current_layer == min_layer {
            start_node_ids.retain(|id| *id != node_id);
        } else if current_layer == max_layer {
            end_node_ids.retain(|id| *id != node_id);
        }

        // Add to aprropriate list if new range is start or end
        if new_layer == min_layer {
            start_node_ids.push(node_id);
        } else if new_layer == max_layer {
            end_node_ids.push(node_id);
        }
    }

    /// Gets time in all active sub-contracts for the given node
    ///
    /// # Arguments
    ///
    /// * `node_id` - Id of node to get time of
    ///
    /// # Returns
    ///
    /// Time left in the queue for given node
    pub fn time_in_queue(&self, node_id: usize) -> u32 {
        let mut total_time = 0;

        for c in self.contracts.iter() {
            for s in c.layers.iter() {
                if s.owner_id == node_id {
                    total_time += s.time_left;
                }
            }
        }

        total_time
    }

    /// Finds the optimal path for a node to form a complete path in the network
    /// Search over all start, end combinations and take minimum
    ///
    /// # Arguments
    ///
    /// * `node` - Node to start from
    /// * `graph` - DAG representation of P2PNetwork
    /// * `order` - Sorted list of nodes in graph
    ///
    /// # Returns
    ///
    /// Gives sequence of node ids that form the shortest path
    pub fn fastest_path_for_node(
        &self,
        node: &P2PNode,
        graph: &DirectedGraph<&P2PNode>,
        order: &Vec<&P2PNode>,
    ) -> Result<Vec<&P2PNode>, String> {
        let mut min_distance = f64::INFINITY;
        let mut shortest_path: Option<Vec<&P2PNode>> = None;
        for start_node in P2PNetwork::start_nodes(self) {
            for end_node in P2PNetwork::end_nodes(self) {
                let start_id = start_node.id;
                let end_id = end_node.id;

                let distance_to_start = NodeParameters::get_latency(&node.params, start_id)
                    .expect("latency to start not defined")
                    .clone();
                let distance_to_end = NodeParameters::get_latency(&end_node.params, node.id)
                    .expect("latency from end not defined")
                    .clone();

                let (path, distance) =
                    P2PNetwork::optimal_path(self, graph, order, start_id, end_id)
                        .expect("path not found");

                let total_distance = distance + distance_to_start as f64 + distance_to_end as f64;

                if total_distance < min_distance {
                    min_distance = total_distance;
                    shortest_path = Some(path);
                }
            }
        }
        match shortest_path {
            Some(path) => Ok(path),
            None => Err(String::from("No path found in the network")),
        }
    }

    /// Finds the optimal path between a min and max layer node
    /// This uses dynamic programming to find optimal path
    /// in a topological sorted graph
    ///
    /// # Arguments
    ///
    /// * `graph` - DAG representation of P2PNetwork
    /// * `order` - Sorted list of nodes in graph
    /// * `start` - Source node
    /// * `end` - Destination node
    pub fn optimal_path(
        &self,
        graph: &DirectedGraph<&P2PNode>,
        order: &Vec<&P2PNode>,
        start: usize,
        end: usize,
    ) -> Result<(Vec<&P2PNode>, f64), String> {
        // Initialize distance and predecessor arrays
        let node_count = order.len();
        let mut distances = vec![f64::INFINITY; node_count]; // Distance from start to each node
        let mut predecessors = vec![usize::MAX; node_count]; // Previous node in optimal path

        // Set distance from start node to itself as zero
        distances[start] = 0.0;

        // Process nodes in topological order
        for current_node in order.iter() {
            let current_id = current_node.id;

            // Skip nodes that haven't been reached yet
            if distances[current_id] == f64::INFINITY {
                continue;
            }

            // Process all neighbors of the current node
            for neighbor in DirectedGraph::neighbors(graph, current_node) {
                let neighbor_id = neighbor.id;

                // Calculate total time to reach neighbor through current node
                let edge_latency =
                    current_node
                        .params
                        .latencies
                        .get(&neighbor_id)
                        .ok_or_else(|| {
                            format!(
                                "Missing latency data for edge {}->{}",
                                current_id, neighbor_id
                            )
                        })?;

                let queue_time = P2PNetwork::time_in_queue(self, neighbor_id);
                let processing_time = neighbor.params.computational_cost;
                let cost = neighbor.price;

                let price_per_ms = 0.5 * cost
                    + 0.5 * ((edge_latency + queue_time + processing_time) as f64 / 50.0);

                // If we found a better path to the neighbor, update it
                if distances[neighbor_id] > distances[current_id] + price_per_ms {
                    distances[neighbor_id] = distances[current_id] + price_per_ms;
                    predecessors[neighbor_id] = current_id;
                }
            }
        }

        // Check if we found a path to the destination
        if distances[end] == f64::INFINITY {
            return Err(format!("No path found from node {} to node {}", start, end));
        }

        // Reconstruct the path by working backwards from end to start
        let mut path = Vec::new();
        let mut current = end;

        while current != usize::MAX {
            // Find the node by ID and add to path
            match P2PNetwork::find_node_by_id(self, current) {
                Some(node) => path.push(node),
                None => return Err(format!("Node with ID {} not found in network", current)),
            }

            current = predecessors[current];
        }

        // Return the path (from end to start)
        Ok((path, distances[end]))
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

    /// Creates a new contract for a given sequence of nodes
    ///
    /// # Arguments
    ///
    /// * `owner` - Creator of the contract
    /// * `nodes` - Sequence of nodes in contract
    ///
    /// # Returns
    ///
    /// New contract in network
    pub fn create_contract(
        &self,
        owner: &P2PNode,
        nodes: Vec<&P2PNode>,
    ) -> Result<Contract, String> {
        let mut sub_contracts = VecDeque::new();

        let start_node = P2PNetwork::start_nodes(self);
        let end_node = P2PNetwork::end_nodes(self);

        if !start_node.contains(&nodes[nodes.len() - 1]) {
            return Err(String::from("first node is not a start node"));
        }

        if !end_node.contains(&nodes[0]) {
            return Err(String::from("last node is not an end node"));
        }

        for i in 0..nodes.len() {
            let dest_idx = if i > 0 { i - 1 } else { usize::MAX };
            let owner_idx = i;
            let source_idx = if i < nodes.len() - 1 {
                i + 1
            } else {
                usize::MAX
            };

            let source_node = if source_idx == usize::MAX {
                P2PNetwork::find_node_by_id(self, owner.id).expect("owner not in network")
            } else {
                &nodes[source_idx]
            };
            let owner_node = &nodes[owner_idx];
            let dest_node = if dest_idx == usize::MAX {
                P2PNetwork::find_node_by_id(self, owner.id).expect("owner not in network")
            } else {
                &nodes[dest_idx]
            };

            let new_subcontract = SubContract::new(
                source_node.id,
                owner_node.id,
                dest_node.id,
                owner_node.params.computational_cost,
                owner_node.price,
            );

            // add to list of subcontracts
            sub_contracts.push_front(new_subcontract);
        }

        Ok(Contract::new(sub_contracts, owner.id))
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

    /// Finds a node in the network by its ID and returns a mutable reference.
    ///
    /// # Arguments
    ///
    /// * `id` - The ID of the node to find
    ///
    /// # Returns
    ///
    /// Option containing a mutable reference to the node if found, None otherwise
    pub fn find_node_by_id_mut(&mut self, id: usize) -> Option<&mut P2PNode> {
        self.nodes.iter_mut().find(|node| node.id == id)
    }
}

/// Represents a sub-contract within a larger contract system
///
/// A `SubContract` defines a relationship between a source and destination entity
/// with a time constraint for completion.
#[derive(Clone, PartialEq, Debug)]
pub struct SubContract {
    /// Identifier for the source entity
    pub source_id: usize,

    /// Owner of the contract doing the computation
    pub owner_id: usize,

    /// Identifier for the destination entity
    pub dest_id: usize,

    /// Time remaining (in seconds) before the sub-contract expires
    pub time_left: u32,

    /// Price to fulfill contract
    pub price: f64,
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
    pub fn new(
        source_id: usize,
        owner_id: usize,
        dest_id: usize,
        time_left: u32,
        price: f64,
    ) -> Self {
        SubContract {
            source_id,
            owner_id,
            dest_id,
            time_left,
            price,
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
}
impl Eq for SubContract {}

/// Represents a multi-layered contract on a blockchain
///
/// A `Contract` consists of multiple sub-contracts that must be fulfilled
/// in sequence. The contract tracks fulfillment status and manages the
/// relationship between all sub-contracts.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Contract {
    owner: usize,
    fulfilled: bool,
    layers: VecDeque<SubContract>,
}

impl Contract {
    /// Creates a new contract with specified sub-contracts
    ///
    /// # Arguments
    ///
    /// * `owner` - Id of owner that owns the contract
    /// * `layers` - A vector of references to `SubContract`s
    ///
    /// # Returns
    ///
    /// A new `Contract` instance
    pub fn new(layers: VecDeque<SubContract>, owner: usize) -> Self {
        Contract {
            owner,
            fulfilled: false,
            layers,
        }
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
        match self.layers.is_empty() {
            true => {
                self.fulfilled = true;
                true
            }
            false => false,
        }
    }

    pub fn is_fulfilled(&self) -> bool {
        self.fulfilled
    }

    pub fn get_owner(&self) -> usize {
        self.owner
    }

    pub fn num_subcontracts(&self) -> usize {
        self.layers.len()
    }

    /// Prints the contract's state and information about its subcontracts
    ///
    /// This method outputs a formatted representation of the contract,
    /// showing its owner ID, fulfillment status, and details of each subcontract.
    pub fn print(&self) {
        println!("Contract Information:");
        println!("  Owner ID: {}", self.owner);
        println!(
            "  Status: {}",
            if self.fulfilled {
                "Fulfilled"
            } else {
                "Pending"
            }
        );
        println!("  Subcontract Count: {}", self.layers.len());

        if !self.layers.is_empty() {
            println!("\nSubcontracts (from first to last):");
            for (i, subcontract) in self.layers.iter().enumerate() {
                println!("  Subcontract #{}:", i + 1);
                println!("    Source ID: {}", subcontract.source_id);
                println!("    Owner ID: {}", subcontract.owner_id);
                println!("    Destination ID: {}", subcontract.dest_id);
                println!("    Time Left: {} ms", subcontract.time_left);
                println!("    Price: {} tokens", subcontract.price);
            }
        }

        println!(
            "\nTotal Contract Value: {} tokens",
            self.layers.iter().map(|sc| sc.price).sum::<f64>()
        );
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct P2PNode {
    pub id: usize,
    pub price: f64,
    pub balance: f64,
    pub params: NodeParameters,
}

impl P2PNode {
    /// Creates a new P2PNode with the given id and parameters.
    ///
    /// # Arguments
    ///
    /// * `id` - A unique identifier for this node
    /// * `price` - Cost of computation with node
    /// * `params` - The parameters that configure this node's behavior
    ///
    /// # Returns
    ///
    /// A new P2PNode instance
    pub fn new(id: usize, params: NodeParameters) -> Self {
        P2PNode {
            id,
            price: 1.0,
            balance: 100.0,
            params,
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
impl Eq for P2PNode {}

// Represents the parameters for a node in a distributed system.
///
/// # Fields
///
/// * `layer_range` - Range of layers node i has
/// * `latencies` - Map of latencies from this node to other nodes, where the key is the node ID
/// * `computational_cost` - Computational cost of the node
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
    /// * `embedding_cost` - Embedding cost value
    ///
    /// # Returns
    ///
    /// A new NodeParameters instance with empty latencies map
    pub fn new(layer_range: u8, computational_cost: u32, embedding_cost: u32) -> Self {
        NodeParameters {
            layer_range,
            latencies: HashMap::new(),
            computational_cost,
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
    pub fn get_latency(&self, destination_node_id: usize) -> Option<&u32> {
        self.latencies.get(&destination_node_id)
    }
}
