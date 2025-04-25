mod backend;

// import p2p network
use backend::graph::DirectedGraph;
use backend::p2p::{make_graph, NodeParameters, P2PNode};

fn main() {
    let node_params_0 = NodeParameters::new(0, 1, 1, 1);
    let node_params_1 = NodeParameters::new(1, 1, 1, 1);
    let node_params_2 = NodeParameters::new(2, 1, 1, 1);
    let node_params_3 = NodeParameters::new(3, 1, 1, 1);

    let node0 = P2PNode::new(0, node_params_0);
    let node1 = P2PNode::new(1, node_params_1);
    let node2 = P2PNode::new(2, node_params_2);
    let node3 = P2PNode::new(3, node_params_3);

    let nodes = vec![node0, node1, node2, node3];

    let graph = make_graph(&nodes);

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
        println!("{}", node.id);
    }
}
