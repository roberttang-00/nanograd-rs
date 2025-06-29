use crate::ops::op_defs::Op;
use ndarray::ArrayD;
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt;
use std::ops::{Add as StdAdd, Sub as StdSub, Mul as StdMul, Div as StdDiv, Neg as StdNeg};
use std::cmp::PartialEq;
use std::convert::Into;

pub type TensorRef = Rc<RefCell<Tensor>>;

#[derive(Clone, Debug)]
pub enum TensorData {
    Scalar(f32),
    Tensor(ArrayD<f32>)
}

impl fmt::Display for TensorData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorData::Scalar(x) => write!(f, "{}", x),
            TensorData::Tensor(arr) => write!(f, "{:?}", arr)
        }
    }
}

impl From<f32> for TensorData {
    fn from(value: f32) -> TensorData{
        TensorData::Scalar(value)
    }
}

impl From<ArrayD<f32>> for TensorData {
    fn from(value: ArrayD<f32>) -> TensorData{
        TensorData::Tensor(value)
    }
}

impl StdNeg for &TensorData {
    type Output = TensorData;

    fn neg(self) -> Self::Output {
        match self {
            TensorData::Scalar(a) => TensorData::Scalar(-*a),
            TensorData::Tensor(a) => TensorData::Tensor(-a)
        }
    }
}

impl PartialEq for &TensorData {
    fn eq(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (TensorData::Scalar(a), TensorData::Scalar(b)) => *a == *b,
            (TensorData::Tensor(a), TensorData::Tensor(b)) => a == b,
            _ => panic!("Trying to compare scalar and array")
        }
    }
}

impl StdAdd for &TensorData {
    type Output = TensorData;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (TensorData::Scalar(a), TensorData::Scalar(b)) => TensorData::Scalar(*a + *b),
            (TensorData::Tensor(a), TensorData::Scalar(b)) => TensorData::Tensor(a + *b),
            (TensorData::Scalar(a), TensorData::Tensor(b)) => TensorData::Tensor(*a + b),
            (TensorData::Tensor(a), TensorData::Tensor(b)) => TensorData::Tensor(a + b)
        }
    }
}

impl StdSub for &TensorData {
    type Output = TensorData;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (TensorData::Scalar(a), TensorData::Scalar(b)) => TensorData::Scalar(*a - *b),
            (TensorData::Tensor(a), TensorData::Scalar(b)) => TensorData::Tensor(a - *b),
            (TensorData::Scalar(a), TensorData::Tensor(b)) => TensorData::Tensor(*a - b),
            (TensorData::Tensor(a), TensorData::Tensor(b)) => TensorData::Tensor(a - b)
        }
    }
}

impl StdMul for &TensorData {
    type Output = TensorData;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (TensorData::Scalar(a), TensorData::Scalar(b)) => TensorData::Scalar(*a * *b),
            (TensorData::Tensor(a), TensorData::Scalar(b)) => TensorData::Tensor(a * *b),
            (TensorData::Scalar(a), TensorData::Tensor(b)) => TensorData::Tensor(*a * b),
            (TensorData::Tensor(a), TensorData::Tensor(b)) => TensorData::Tensor(a * b)
        }
    }
}


impl StdDiv for &TensorData {
    type Output = TensorData;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (TensorData::Scalar(a), TensorData::Scalar(b)) => TensorData::Scalar(*a / *b),
            (TensorData::Tensor(a), TensorData::Scalar(b)) => TensorData::Tensor(a / *b),
            (TensorData::Scalar(a), TensorData::Tensor(b)) => TensorData::Tensor(*a / b),
            (TensorData::Tensor(a), TensorData::Tensor(b)) => TensorData::Tensor(a / b)
        }
    }
}


#[derive(Clone)]
pub struct Tensor {
    pub data: TensorData,
    pub grad: Option<TensorData>,
    pub requires_grad: bool,
    pub grad_fn: Option<Rc<dyn Op>>,
    pub parents: Vec<TensorRef>
}

impl Tensor {
    pub fn new<T: Into<TensorData>>(data: T, requires_grad: bool) -> TensorRef {
        Rc::new(RefCell::new(Tensor {
            data: data.into(),
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
                tensor.grad = Some(match &tensor.data {
                    TensorData::Scalar(_) => TensorData::Scalar(1.0),
                    TensorData::Tensor(x) => TensorData::Tensor(ArrayD::ones(x.raw_dim()))
                });
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
                            Some(existing) => existing + &parent_grad,
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