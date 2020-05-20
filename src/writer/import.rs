use std::fmt;

pub fn write_imports(f: &mut fmt::Formatter<'_>) -> fmt::Result {
    for import in &IMPORTS {
        write!(f, "{}", import)?;
    }
    Ok(())
}

const IMPORTS: [PyImport; 3] = [
    PyImport::new("torch"),
    PyImport::new_alias("torch.nn", "nn"),
    PyImport::new("n3"),
];

struct PyImport<'a> {
    pub name: &'a str,
    pub from: Option<&'a str>,
    pub alias: Option<&'a str>,
}

impl<'a> PyImport<'a> {
    const fn new(name: &'a str) -> Self {
        Self {
            name,
            from: None,
            alias: None,
        }
    }

    const fn new_alias(name: &'a str, alias: &'a str) -> Self {
        Self {
            name,
            from: None,
            alias: Some(alias),
        }
    }
}

impl<'a> fmt::Display for PyImport<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(from) = &self.from {
            write!(f, " from {}", from)?;
        }
        write!(f, "import {}", &self.name)?;
        if let Some(alias) = &self.alias {
            write!(f, " as {}", alias)?;
        }
        writeln!(f)?;
        Ok(())
    }
}
