mod backend;

// import p2p network
use backend::graph::DirectedGraph;
use backend::p2p::{Contract, NodeParameters, P2PNetwork, P2PNode};

fn main() {
    let mut node_params_0 = NodeParameters::new(0, 1, 1);
    let mut node_params_1 = NodeParameters::new(1, 1, 1);
    let mut node_params_2 = NodeParameters::new(2, 1, 1);
    let mut node_params_3 = NodeParameters::new(3, 1, 1);
    let mut node_params_4 = NodeParameters::new(1, 2, 1);

    NodeParameters::set_latency(&mut node_params_0, 1, 5);
    NodeParameters::set_latency(&mut node_params_0, 0, 0);
    NodeParameters::set_latency(&mut node_params_4, 0, 3);
    NodeParameters::set_latency(&mut node_params_1, 2, 9);
    NodeParameters::set_latency(&mut node_params_2, 4, 2);
    NodeParameters::set_latency(&mut node_params_3, 0, 7);
    NodeParameters::set_latency(&mut node_params_4, 2, 11);
    NodeParameters::set_latency(&mut node_params_0, 3, 11);

    let node0 = P2PNode::new(0, node_params_0);
    let node1 = P2PNode::new(1, node_params_1);
    let node2 = P2PNode::new(2, node_params_2);
    let node3 = P2PNode::new(4, node_params_3);
    let node4 = P2PNode::new(3, node_params_4);

    let nodes = vec![node0, node1, node2, node3, node4];

    let mut network = P2PNetwork::from_nodes(nodes);

    let graph = P2PNetwork::make_graph(&network);

    println!(
        "Min Layer: {} Make Layer: {}",
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
}
