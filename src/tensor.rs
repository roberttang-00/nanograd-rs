use crate::ops::op_defs::Op;
use ndarray::ArrayD;
use std::rc::Rc;
use std::cell::RefCell;

pub type TensorRef = Rc<RefCell<Tensor>>;

#[derive(Clone)]
pub struct Tensor {
    pub data: ArrayD<f32>,
    pub grad: Option<ArrayD<f32>>,
    pub requires_grad: bool,
    pub grad_fn: Option<Rc<dyn Op>>,
    pub parents: Vec<TensorRef>
}

impl Tensor {
    pub fn new(data: ArrayD<f32>, requires_grad: bool) -> TensorRef {
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
                tensor.grad = Some(ArrayD::ones(tensor.data.raw_dim()));
            }
        }
        println!("Starting backward, root grad: {:?}", self_.borrow().grad);

        let mut stack = vec![self_.clone()];
        while let Some(current) = stack.pop() {
            let grad = current.borrow().grad.clone().unwrap();
            println!("Processing tensor with grad: {}", grad);
            
            let (grad_fn, parents) = {
                let current_ref = current.borrow();
                (current_ref.grad_fn.clone(), current_ref.parents.clone())
            };
            println!("Has grad_fn: {}, num parents: {}", grad_fn.is_some(), parents.len());

            if let Some(op) = grad_fn {
                println!("Calling backward on op: {:?}", op);
                let grads = op.backward(&current, &grad);
                println!("Computed gradients: {:?}", grads);
                
                for (i, (parent, parent_grad)) in parents.iter().zip(grads).enumerate() {
                    if parent.borrow().requires_grad {
                        println!("Updating parent {}: old_grad={:?}, new_grad={}", 
                        i, parent.borrow().grad, parent_grad);

                        let mut p = parent.borrow_mut();
                        p.grad = Some(match &p.grad {
                            Some(existing) => existing + parent_grad,
                            None => parent_grad,
                        });
                        println!("Parent {} final grad: {:?}", i, p.grad);
                        drop(p);
                        stack.push(parent.clone())
                    }
                }
            }
        }
        println!("Backward complete");
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