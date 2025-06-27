use crate::ops::op_defs::Op;
use std::rc::Rc;
use std::cell::RefCell;

pub type TensorRef = Rc<RefCell<Tensor>>;

#[derive(Clone)]
pub struct Tensor {
    pub data: f32,
    pub grad: Option<f32>,
    pub requires_grad: bool,
    pub grad_fn: Option<Rc<dyn Op>>,
    pub parents: Vec<TensorRef>
}

impl Tensor {
    pub fn new(data: f32, requires_grad: bool) -> TensorRef {
        Rc::new(RefCell::new(Tensor {
            data,
            grad: None,
            requires_grad,
            grad_fn: None,
            parents: vec![]
        }))
    }

    pub fn backward(self_: &TensorRef) {
        {
            let mut tensor = self_.borrow_mut();
            if tensor.grad.is_none() {
                tensor.grad = Some(1.0);
            }
        }

        let mut stack = vec![self_.clone()];
        while let Some(current) = stack.pop() {
            let grad = current.borrow().grad.clone().unwrap();
            
            let (grad_fn, parents) = {
                let current_ref = current.borrow();
                (current_ref.grad_fn.clone(), current_ref.parents.clone())
            };

            if let Some(op) = grad_fn {
                let grads = op.backward(&current, &grad);
                for (parent, parent_grad) in parents.iter().zip(grads) {
                    let mut p = parent.borrow_mut();
                    p.grad = Some(match p.grad {
                        Some(existing) => existing + parent_grad,
                        None => parent_grad,
                    });
                    stack.push(parent.clone());
                }
            }
        }
    }
}

pub trait TensorOps {
    fn backward(&self);
}

impl TensorOps for TensorRef {
    fn backward(&self) {
        Tensor::backward(self);
    }
}