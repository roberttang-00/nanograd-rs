use crate::tensor::TensorRef;
use std::fmt::Debug;

pub trait Op: Debug {
    fn backward(&self, output: &TensorRef, grad_output: &f32) -> Vec<f32>;
}

#[derive(Debug)]
pub struct Add;

#[derive(Debug)]
pub struct Mul;

impl Op for Add {
    fn backward(&self, _output: &TensorRef, grad_output: &f32) -> Vec<f32> {
        vec![grad_output.clone(), grad_output.clone()]
    }
}

impl Op for Mul {
    fn backward(&self, output: &TensorRef, grad_output: &f32) -> Vec<f32> {
        let output_borrow = output.borrow();
        let lhs = &output_borrow.parents[0].borrow().data;
        let rhs = &output_borrow.parents[1].borrow().data;
        vec![
            *grad_output * *rhs,
            *grad_output * *lhs
        ]
    }
}