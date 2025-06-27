use crate::tensor::{Tensor, TensorRef};
use crate::ops::op_defs::{Op, Add, Sub, Mul, Div};
use ndarray::ArrayD;
use std::rc::Rc;

impl Op for Add {
    fn forward(&self, inputs: &[&TensorRef]) -> ArrayD<f32> {
        &inputs[0].borrow().data + &inputs[1].borrow().data
    }

    fn backward(&self, _output: &TensorRef, grad_output: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        vec![grad_output.clone(), grad_output.clone()]
    }

    fn name(&self) -> &'static str { "Add" }
}

impl Op for Sub {
    fn forward(&self, inputs: &[&TensorRef]) -> ArrayD<f32> {
        &inputs[0].borrow().data - &inputs[1].borrow().data
    }

    fn backward(&self, _output: &TensorRef, grad_output: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        vec![grad_output.clone(), -grad_output.clone()]
    }

    fn name(&self) -> &'static str { "Sub" }
}

impl Op for Mul {
    fn forward(&self, inputs: &[&TensorRef]) -> ArrayD<f32> {
        &inputs[0].borrow().data * &inputs[1].borrow().data
    }

    fn backward(&self, output: &TensorRef, grad_output: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let output_borrow = output.borrow();
        let lhs = &output_borrow.parents[0].borrow().data;
        let rhs = &output_borrow.parents[1].borrow().data;
        vec![
            grad_output * rhs, // dL/da = dL/dz * b
            grad_output * lhs  // dL/db = dL/dz * a
        ]
    }

    fn name(&self) -> &'static str { "Mul" }
}

impl Op for Div {
    fn forward(&self, inputs: &[&TensorRef]) -> ArrayD<f32> {
        let denominator = &inputs[1].borrow().data;
        if denominator.iter().any(|&x| x == 0.0) {
            panic!("division by zero");
        }
        &inputs[0].borrow().data / denominator
    }

    fn backward(&self, output: &TensorRef, grad_output: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let output_borrow = output.borrow();
        let lhs = &output_borrow.parents[0].borrow().data;
        let rhs = &output_borrow.parents[1].borrow().data;

        if rhs.iter().any(|&x| x == 0.0) {
            panic!("division by zero");
        }

        let dzda = 1.0 / rhs;                    // dz/da = 1/b
        let dzdb = -lhs / (rhs * rhs);           // dz/db = -a/b^2
        vec![
            grad_output * dzda, // dL/da = dL/dz * dz/da
            grad_output * dzdb  // dL/db = dL/dz * dz/db
        ]
    }

    fn name(&self) -> &'static str { "Div" }
}

fn apply_binary_op(a: &TensorRef, b: &TensorRef, op: Rc<dyn Op>) -> TensorRef {
    let data = op.forward(&[a, b]);
    let requires_grad = a.borrow().requires_grad || b.borrow().requires_grad;
    let result = Tensor::new(data, requires_grad);

    if requires_grad {
        result.borrow_mut().parents = vec![a.clone(), b.clone()];
        result.borrow_mut().grad_fn = Some(op);
    }

    result
}

pub fn add(a: &TensorRef, b: &TensorRef) -> TensorRef {
    apply_binary_op(a, b, Rc::new(Add))
}

pub fn sub(a: &TensorRef, b: &TensorRef) -> TensorRef {
    apply_binary_op(a, b, Rc::new(Sub))
}

pub fn mul(a: &TensorRef, b: &TensorRef) -> TensorRef {
    apply_binary_op(a, b, Rc::new(Mul))
}

pub fn div(a: &TensorRef, b: &TensorRef) -> TensorRef {
    apply_binary_op(a, b, Rc::new(Div))
}