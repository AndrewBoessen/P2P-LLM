mod backend;

// import p2p network
use backend::graph::DirectedGraph;
use backend::p2p::{NodeParameters, P2PNetwork, P2PNode};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use std::{thread, time};

fn main() {
    //let mut rng = rand::rng();
    let mut rng = ChaCha12Rng::seed_from_u64(42);
    let node_count = vec![2, 4, 2, 4];
    let mut node_id = 0;

    let mut nodes: Vec<P2PNode> = Vec::new();

    for (layer, num) in node_count.iter().enumerate() {
        for _ in 0..num.clone() {
            let new_params = NodeParameters::new(layer as u8, rng.random_range(1..50), 1);
            let new_node = P2PNode::new(node_id, new_params);
            nodes.push(new_node);
            node_id += 1;
        }
    }

    let total_nodes = nodes.len();
    for node in nodes.iter_mut() {
        for dest_id in 0..total_nodes {
            let params = &mut node.params;
            NodeParameters::set_latency(params, dest_id, rng.random_range(1..10));
        }
    }

    let mut network = P2PNetwork::from_nodes(nodes);

    let mut iter = 0;
    loop {
        // build graph
        let graph = P2PNetwork::make_graph(&network);
        let order = DirectedGraph::topological_sort(&graph)
            .expect("network has a cycle, graph could not be built");

        for node in order.iter() {
            println!("{}", node.price / node.params.computational_cost as f64);
        }

        let nodes_to_process = P2PNetwork::nodes_without_contracts(&network);

        // Create new contracts
        let mut new_contracts = Vec::new();
        for node_info in nodes_to_process {
            if let Ok(path) = P2PNetwork::fastest_path_for_node(&network, node_info, &graph, &order)
            {
                if let Ok(contract) = P2PNetwork::create_contract(&network, node_info, path) {
                    new_contracts.push(contract);
                }
            }
        }

        // work on contracts
        let node_to_update = rng.random_range(0..network.nodes.len());
        P2PNetwork::update_network(&mut network, 1, iter, 1000, node_to_update);

        // Add all new contracts at once
        network.contracts.extend(new_contracts);

        iter += 1;

        thread::sleep(time::Duration::from_millis(10));
    }
}
