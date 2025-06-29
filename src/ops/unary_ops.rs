use crate::tensor::*;
use crate::ops::op_defs::{Op, Neg, Abs, ReLU};
use std::rc::Rc;

impl Op for Neg {
    fn forward(&self, inputs: &[&TensorRef]) -> TensorData {
        -&inputs[0].borrow().data
    }

    fn backward(&self, _output: &TensorRef, grad_output: &TensorData) -> Vec<TensorData> {
        vec![-&grad_output.clone()]
    }

    fn name(&self) -> &'static str { "Neg" }
}

impl Op for Abs {
    fn forward(&self, inputs: &[&TensorRef]) -> TensorData {
        let x = &inputs[0].borrow().data;

        match x {
            TensorData::Scalar(x) => TensorData::Scalar((*x).abs()),
            TensorData::Tensor(arr) => TensorData::Tensor(arr.map(|x| (*x).abs()))
        }
    }

    fn backward(&self, _output: &TensorRef, grad_output: &TensorData) -> Vec<TensorData> {
        match grad_output {
            TensorData::Scalar(x) => vec![TensorData::Scalar((*x) * (*x).signum())],
            TensorData::Tensor(arr) => vec![TensorData::Tensor(arr.map(|x| (*x) * (*x).signum()))]
        }
    }

    fn name(&self) -> &'static str { "Abs" }
}

impl Op for ReLU {
    fn forward(&self, inputs: &[&TensorRef]) -> TensorData {
        let x = &inputs[0].borrow().data;

        match x {
            TensorData::Scalar(x) => TensorData::Scalar((*x).max(0.0f32)),
            TensorData::Tensor(arr) => TensorData::Tensor(arr.map(|x| (*x).max(0.0f32)))
        }
    }

    fn backward(&self, _output: &TensorRef, grad_output: &TensorData) -> Vec<TensorData> {
        match grad_output {
            TensorData::Scalar(x) => vec![TensorData::Scalar((*x) * ((*x) > 0.0f32) as u8 as f32)],
            TensorData::Tensor(arr) => vec![TensorData::Tensor(arr.map(|x| (*x) * ((*x) > 0.0f32) as u8 as f32))]
        }
    }

    fn name(&self) -> &'static str { "ReLU" }
}


fn apply_unary_op(a: &TensorRef, op: Rc<dyn Op>) -> TensorRef {
    let data = op.forward(&[a]);
    let requires_grad = a.borrow().requires_grad;
    let result = Tensor::new(data, requires_grad);

    if requires_grad {
        result.borrow_mut().parents = vec![a.clone()];
        result.borrow_mut().grad_fn = Some(op);
    }

    result
}

pub fn neg(a: &TensorRef) -> TensorRef {
    apply_unary_op(a, Rc::new(Neg))
}

pub fn abs(a: &TensorRef) -> TensorRef {
    apply_unary_op(a, Rc::new(Abs))
}

pub fn relu(a: &TensorRef) -> TensorRef {
    apply_unary_op(a, Rc::new(ReLU))
}