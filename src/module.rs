use std::collections::BTreeMap;
use std::fmt;

use crate::error::CompileError;
use crate::guide::Guide;
use crate::shape::Shapes;
use crate::writer::{self, indent::*};

use n3_core::{GraphId, GraphIdArg, Variable};

#[derive(Debug)]
pub struct TorchMainModule<'a>(TorchModule<'a>);

impl<'a> TorchMainModule<'a> {
    pub(crate) fn new(module: TorchModule<'a>) -> Self {
        Self(module)
    }
}

impl<'a> fmt::Display for TorchMainModule<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writer::write_imports(f)?;
        write!(f, "\n\n{}", &self.0)?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct TorchModule<'a> {
    pub(crate) name: &'a str,
    pub(crate) variables: BTreeMap<String, &'a Variable>,
    pub(crate) nodes: BTreeMap<GraphId, TorchSubModule<'a>>,
    pub(crate) input: Option<Shapes>,
    pub(crate) output: Option<Shapes>,
    pub(crate) args: Option<&'a [GraphIdArg]>,
    pub(crate) guide: Guide,
}

impl<'a> TorchModule<'a> {
    pub(crate) fn new_extern(
        name: &'a str,
        variables: BTreeMap<String, &'a Variable>,
    ) -> Result<Self, CompileError> {
        let guide = Guide::find_extern_guide(name)?;

        Ok(TorchModule {
            name,
            variables,
            nodes: Default::default(),
            input: None,
            output: None,
            args: None,
            guide,
        })
    }
}

impl<'a> fmt::Display for TorchModule<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "class {}(nn.Module):", self.name)?;

        // constructor
        {
            write!(f, "{}def __init__(self", INDENT1)?;
            for (name, var) in &self.variables {
                write!(f, ", {}: {}", name, writer::type_to_str(&var.ty))?;
                write!(f, " = ")?;
                writer::write_value(f, var.value.as_ref())?;
            }
            writeln!(f, "):")?;

            writeln!(f, "{}super().__init__()", INDENT2)?;
            for name in self.variables.keys() {
                writeln!(f, "{}self.{} = {}", INDENT2, name, name)?;
            }
            writeln!(f)?;

            for (id, node) in &self.nodes {
                write!(f, "{}self.node_", INDENT2)?;
                writer::write_id(f, id)?;
                writeln!(f, " = {}", node)?;
            }
            writeln!(f)?;
        }

        // forward
        {
            write!(f, "{}def forward(self, ", INDENT1)?;
            if let Some(args) = self.args {
                for arg in args {
                    write!(f, "x_")?;
                    writer::write_id_arg(f, arg)?;
                }
            }
            writeln!(f, "):")?;

            for (id, node) in &self.nodes {
                write!(f, "{}", INDENT2)?;
                writer::write_id_args(f, id, node.0.output.as_ref().unwrap().0.len())?;

                write!(f, " = self.node_")?;
                writer::write_id(f, id)?;

                write!(f, "(")?;
                for input in node.0.args.unwrap() {
                    write!(f, "x_")?;
                    writer::write_id_arg(f, input)?;
                }
                writeln!(f, ")")?;
            }

            write!(f, "{}return ", INDENT2)?;
            let last_id = self.nodes.keys().rev().next().unwrap();
            writer::write_id_args(f, last_id, self.output.as_ref().unwrap().0.len())?;
            writeln!(f)
        }
    }
}

#[derive(Debug)]
pub struct TorchSubModule<'a>(pub TorchModule<'a>);

impl<'a> fmt::Display for TorchSubModule<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.guide.write_fmt(f, &self.0)
    }
}
