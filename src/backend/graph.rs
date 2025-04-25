use std::collections::{HashMap, HashSet};

/// A directed graph implementation using adjacency lists
pub struct DirectedGraph<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    /// Maps each node to its set of outgoing edges
    adjacency_list: HashMap<T, HashSet<T>>,
}

impl<T> DirectedGraph<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    /// Creates a new empty directed graph
    pub fn new() -> Self {
        DirectedGraph {
            adjacency_list: HashMap::new(),
        }
    }

    /// Adds a node to the graph if it doesn't already exist
    pub fn add_node(&mut self, node: T) {
        self.adjacency_list.entry(node).or_insert_with(HashSet::new);
    }

    /// Adds a directed edge from source to destination
    /// If the nodes don't exist yet, they are automatically added
    pub fn add_edge(&mut self, source: T, destination: T) {
        // Add source and destination nodes if they don't exist
        self.add_node(source.clone());
        self.add_node(destination.clone());

        // Add the edge
        self.adjacency_list
            .get_mut(&source)
            .unwrap()
            .insert(destination);
    }

    /// Returns true if the graph contains the node
    pub fn has_node(&self, node: &T) -> bool {
        self.adjacency_list.contains_key(node)
    }

    /// Returns true if there is a direct edge from source to destination
    pub fn has_edge(&self, source: &T, destination: &T) -> bool {
        match self.adjacency_list.get(source) {
            Some(neighbors) => neighbors.contains(destination),
            None => false,
        }
    }

    /// Returns a vector of all nodes in the graph
    pub fn nodes(&self) -> Vec<T> {
        self.adjacency_list.keys().cloned().collect()
    }

    /// Returns a vector of all neighbors of a node
    pub fn neighbors(&self, node: &T) -> Vec<T> {
        match self.adjacency_list.get(node) {
            Some(neighbors) => neighbors.iter().cloned().collect(),
            None => Vec::new(),
        }
    }
}
