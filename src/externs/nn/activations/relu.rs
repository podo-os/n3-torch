#[derive(Debug)]
pub struct ReLU;

impl super::super::super::base::ExternModuleImpl for ReLU {
    fn prefix(&self) -> &str {
        "nn"
    }
}
