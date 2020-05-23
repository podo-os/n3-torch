use std::collections::{BTreeMap, HashMap};

use crate::error::CompileError;
use crate::guide::Guide;
use crate::module::{TorchMainModule, TorchModule, TorchSubModule};
use crate::shape::Shapes;

use heck::CamelCase;
use n3_core::{Graph, GraphRoot, Node, UseOrigin, Variable};

pub fn compile_graph<'a, 'b>(
    root: &'a mut GraphRoot,
    name: &'b str,
    origin: UseOrigin,
) -> Result<TorchMainModule<'a>, CompileError>
where
    'b: 'a,
{
    let graph = root.find_graph(name, origin)?;
    let module = graph.compile(name)?;
    Ok(TorchMainModule::new(module))
}

impl<'a> Compile<'a> for Graph {
    type Args = &'a str;
    type Output = TorchModule<'a>;

    fn compile(&'a self, name: Self::Args) -> Result<Self::Output, CompileError> {
        let variables = self.get_variables().compile(())?;

        if self.is_extern() {
            TorchModule::new_extern(name, variables)
        } else {
            let mut inputs = self.get_shapes().into_iter().map(|(_, v)| v).peekable();
            let nodes: BTreeMap<_, _> = self
                .get_nodes()
                .iter()
                .skip(1)
                .map(|(id, node)| {
                    let input = inputs.next().unwrap().into();
                    let output = inputs.peek().unwrap().clone().into();
                    let sub = node.compile((input, output))?;
                    Ok((*id, sub))
                })
                .collect::<Result<_, CompileError>>()?;

            let (_, input) = self.get_shapes().into_iter().next().unwrap();
            let input = input.into();

            let (_, output) = self.get_shapes().into_iter().rev().next().unwrap();
            let output = output.into();

            let args = nodes.iter().next().unwrap().1 .0.args;

            Ok(TorchModule {
                name,
                variables,
                nodes,
                input: Some(input),
                output: Some(output),
                args,
                guide: Guide::NonExtern,
            })
        }
    }
}

impl<'a> Compile<'a> for Node {
    type Args = (Shapes, Shapes);
    type Output = TorchSubModule<'a>;

    fn compile(&'a self, (input, output): Self::Args) -> Result<Self::Output, CompileError> {
        let name = &self.name;

        let args = &self.inputs;

        let mut module = self.graph.as_ref().unwrap().compile(name)?;
        module.input = Some(input);
        module.output = Some(output);
        module.args = Some(args);

        Ok(TorchSubModule(module))
    }
}

impl<'a> Compile<'a> for HashMap<String, Variable> {
    type Args = ();
    type Output = BTreeMap<String, &'a Variable>;

    fn compile(&'a self, _: Self::Args) -> Result<Self::Output, CompileError> {
        Ok(self.iter().map(|(k, v)| (k.to_camel_case(), v)).collect())
    }
}

pub trait Compile<'a> {
    type Args;
    type Output;

    fn compile(&'a self, args: Self::Args) -> Result<Self::Output, CompileError>;
}
