use crate::tensor::*;
use std::fmt::Debug;

pub trait Op: Debug {
    fn forward(&self, inputs: &[&TensorRef]) -> TensorData;
    fn backward(&self, output: &TensorRef, grad_output: &TensorData) -> Vec<TensorData>;
    fn name(&self) -> &'static str { "PrimitiveOp "}
}

#[derive(Debug)]
pub struct Add;

#[derive(Debug)]
pub struct Mul;

#[derive(Debug)]
pub struct Sub;

#[derive(Debug)]
pub struct Div;