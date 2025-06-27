use crate::tensor::{Tensor, TensorRef};
use crate::ops::op_defs::Mul;
use std::rc::Rc;

pub fn mul(a: &TensorRef, b: &TensorRef) -> TensorRef {
    let data = &a.borrow().data * &b.borrow().data;
    let requires_grad = a.borrow().requires_grad || b.borrow().requires_grad;
    let result = Tensor::new(data, requires_grad);

    if requires_grad {
        result.borrow_mut().parents = vec![a.clone(), b.clone()];
        result.borrow_mut().grad_fn = Some(Rc::new(Mul));
    }

    result
}