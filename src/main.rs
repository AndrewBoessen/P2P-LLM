mod backend;

// import p2p network
use backend::graph::DirectedGraph;
use backend::p2p::{NodeParameters, P2PNetwork, P2PNode};

fn main() {
    let mut node_params_0 = NodeParameters::new(0, 1, 1);
    let mut node_params_1 = NodeParameters::new(1, 1, 1);
    let mut node_params_2 = NodeParameters::new(2, 1, 1);
    let mut node_params_3 = NodeParameters::new(3, 1, 1);

    NodeParameters::set_latency(&mut node_params_0, 1, 5);
    NodeParameters::set_latency(&mut node_params_1, 1, 9);
    NodeParameters::set_latency(&mut node_params_2, 1, 2);
    NodeParameters::set_latency(&mut node_params_3, 1, 7);

    let node0 = P2PNode::new(0, node_params_0);
    let node1 = P2PNode::new(1, node_params_1);
    let node2 = P2PNode::new(2, node_params_2);
    let node3 = P2PNode::new(3, node_params_3);

    let nodes = vec![node0, node1, node2, node3];

    let network = P2PNetwork::from_nodes(nodes);

    let graph = P2PNetwork::make_graph(&network);

    println!(
        "Min Layer: {} Make Layer: {}",
        network.min_layer, network.max_layer
    );

    let min_nodes = P2PNetwork::start_nodes(&network);

    for node in min_nodes {
        println!("Start Node: {}", node.id);
    }
    let max_nodes = P2PNetwork::end_nodes(&network);
    for node in max_nodes {
        println!("End Node: {}", node.id);
    }

    let nodes_in_graph = DirectedGraph::nodes(&graph);

    for node in nodes_in_graph {
        let neighbors = DirectedGraph::neighbors(&graph, &node);
        println!("Node ID: {} Layer: {}", node.id, node.params.layer_range);
        for n in neighbors {
            println!("Neighbor {}", n.id);
        }
    }

    let sorted_nodes = match DirectedGraph::topological_sort(&graph) {
        Ok(nodes) => nodes,
        Err(e) => panic!("Error sorting nodes: {}", e),
    };

    println!("Sorted Nodes");
    for node in sorted_nodes {
        let l = NodeParameters::get_latency(&node.params, &1).expect("latency not defined");
        println!("Node ID: {} Latency to 1: {}", node.id, l);
    }
}
