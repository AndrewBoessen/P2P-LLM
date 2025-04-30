mod backend;

// import p2p network
use backend::graph::DirectedGraph;
use backend::p2p::{Contract, NodeParameters, P2PNetwork, P2PNode};

use rand::Rng;
fn main() {
    let mut rng = rand::rng();
    let node_count = vec![2, 3, 1, 3];
    let mut node_id = 0;

    let mut nodes: Vec<P2PNode> = Vec::new();

    for (layer, num) in node_count.iter().enumerate() {
        for _ in 0..num.clone() {
            let new_params = NodeParameters::new(layer as u8, rng.random_range(1..30), 1);
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

    let graph = P2PNetwork::make_graph(&network);

    println!(
        "Min Layer: {} Max Layer: {}",
        network.min_layer, network.max_layer
    );

    let min_nodes = P2PNetwork::start_nodes(&network);

    for node in min_nodes.iter() {
        println!("Start Node: {}", node.id);
    }
    let max_nodes = P2PNetwork::end_nodes(&network);
    for node in max_nodes.iter() {
        println!("End Node: {}", node.id);
    }

    let nodes_in_graph = DirectedGraph::nodes(&graph);

    for node in nodes_in_graph {
        let neighbors = DirectedGraph::neighbors(&graph, &node);
        println!("Node ID: {} Layer: {}", node.id, node.params.layer_range);
        println!(
            "Market Share: {}",
            P2PNetwork::market_share(&network, node, node.price)
        );
        for n in neighbors {
            println!("Neighbor {}", n.id);
        }
    }

    let sorted_nodes = match DirectedGraph::topological_sort(&graph) {
        Ok(nodes) => nodes,
        Err(e) => panic!("Error sorting nodes: {}", e),
    };

    let first_node = match P2PNetwork::find_node_by_id(&network, 0) {
        Some(n) => n,
        None => panic!("Node 0 not found"),
    };

    let fastest_path =
        P2PNetwork::fastest_path_for_node(&network, first_node, &graph, &sorted_nodes)
            .expect("no path found");

    let contract = P2PNetwork::create_contract(&network, 0, fastest_path)
        .expect("contract could not be created");

    Contract::print(&contract);

    network.contracts.push(contract);

    let ids = P2PNetwork::update_network(&mut network, 1);

    for id in ids {
        println!("ID: {}", id);
    }
}
