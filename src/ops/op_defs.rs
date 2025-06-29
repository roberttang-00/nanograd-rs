use crate::tensor::*;
use std::fmt::Debug;

pub trait Op: Debug {
    fn forward(&self, inputs: &[&TensorRef]) -> TensorData;
    fn backward(&self, output: &TensorRef, grad_output: &TensorData) -> Vec<TensorData>;
    fn name(&self) -> &'static str { "PrimitiveOp "}
}

// Unary Ops

#[derive(Debug)]
pub struct Neg;

#[derive(Debug)]
pub struct Abs;

#[derive(Debug)]
pub struct ReLU;

// Binary Ops

#[derive(Debug)]
pub struct Add;

#[derive(Debug)]
pub struct Mul;

#[derive(Debug)]
pub struct Sub;

#[derive(Debug)]
pub struct Div;