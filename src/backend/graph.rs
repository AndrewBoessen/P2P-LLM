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

    /// Performs a topological sort of the graph
    /// Returns a Result containing either:
    /// - Ok(Vec<T>): A topologically sorted vector of nodes
    /// - Err(String): An error message if the graph is not a DAG (contains cycles)
    pub fn topological_sort(&self) -> Result<Vec<T>, String> {
        // Track visited nodes and recursion stack to detect cycles
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut result = Vec::new();

        // Define a recursive DFS function for topological sort
        fn dfs<T: Eq + std::hash::Hash + Clone>(
            graph: &DirectedGraph<T>,
            node: &T,
            visited: &mut HashSet<T>,
            rec_stack: &mut HashSet<T>,
            result: &mut Vec<T>,
        ) -> Result<(), String> {
            // Mark the current node as visited and add to recursion stack
            visited.insert(node.clone());
            rec_stack.insert(node.clone());

            // Visit all neighbors
            for neighbor in graph.neighbors(node) {
                // If neighbor is in recursion stack, we found a cycle
                if rec_stack.contains(&neighbor) {
                    return Err(format!("Graph contains a cycle"));
                }

                // If not visited, recursively visit
                if !visited.contains(&neighbor) {
                    if let Err(e) = dfs(graph, &neighbor, visited, rec_stack, result) {
                        return Err(e);
                    }
                }
            }

            // Remove the node from recursion stack and add it to result
            rec_stack.remove(node);
            result.push(node.clone());

            Ok(())
        }

        // Perform DFS on each unvisited node
        for node in self.nodes() {
            if !visited.contains(&node) {
                if let Err(e) = dfs(self, &node, &mut visited, &mut rec_stack, &mut result) {
                    return Err(e);
                }
            }
        }

        // Reverse the result to get the correct topological order
        result.reverse();

        Ok(result)
    }
}
