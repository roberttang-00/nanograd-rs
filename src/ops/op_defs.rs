use crate::tensor::{Tensor, TensorRef};
use ndarray::ArrayD;
use std::fmt::Debug;
use std::rc::Rc;

pub trait Op: Debug {
    fn forward(&self, inputs: &[&TensorRef]) -> ArrayD<f32>;
    fn backward(&self, output: &TensorRef, grad_output: &ArrayD<f32>) -> Vec<ArrayD<f32>>;
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