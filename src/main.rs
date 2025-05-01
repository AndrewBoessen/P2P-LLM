mod backend;
mod gui;

// import p2p network
use backend::p2p::{NodeParameters, P2PNetwork, P2PNode};
use gui::sim::App;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use std::{io, process};

fn main() -> io::Result<()> {
    // Set up random number generator with deterministic seed
    let mut rng = ChaCha12Rng::seed_from_u64(42);

    // Define network structure with nodes per layer
    let node_count = vec![2, 4, 2, 4];
    let mut node_id = 0;
    let mut nodes: Vec<P2PNode> = Vec::new();

    // Create nodes for each layer
    for (layer, num) in node_count.iter().enumerate() {
        for _ in 0..*num {
            // Random computational cost between 1-50
            let comp_cost = rng.gen_range(1..50);
            let new_params = NodeParameters::new(layer as u8, comp_cost, 1);
            let new_node = P2PNode::new(node_id, new_params);
            nodes.push(new_node);
            node_id += 1;
        }
    }

    // Set up random latencies between nodes
    let total_nodes = nodes.len();
    for node in nodes.iter_mut() {
        for dest_id in 0..total_nodes {
            let latency = rng.gen_range(1..10);
            node.params.set_latency(dest_id, latency);
        }
    }

    // Create the P2P network from nodes
    let network = P2PNetwork::from_nodes(nodes);

    // Initialize and run the TUI application
    let mut app = App::new(network);
    if let Err(err) = app.run_simulation() {
        eprintln!("Application error: {}", err);
        process::exit(1);
    }

    Ok(())
}

