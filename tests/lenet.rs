use n3_core::*;

#[test]
fn build_lenet() {
    let mut root = GraphRoot::with_path("models").unwrap();

    let graph = root.find_graph("LeNet", UseOrigin::Local).unwrap();

    dbg!(graph.get_shapes());
}
