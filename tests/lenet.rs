use std::fs;
use std::io::Write;

use n3_core::*;
use n3_torch_core::compile_graph;

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn compile_lenet() {
    let mut root = GraphRoot::with_path("models").unwrap();

    let module = compile_graph(&mut root, "LeNet", UseOrigin::Local).unwrap();

    let mut file = fs::File::create("out-lenet.py").unwrap();
    write!(
        &mut file,
        "#! auto-generated file written by N3\n\n{}",
        module
    )
    .unwrap();
}
