use crate::tensor::*;
use crate::ops::op_defs::{Op, Sum, Mean};
use ndarray::{ArrayD, Axis, IxDyn};
use std::rc::Rc;

impl Op for Sum {
    fn forward(&self, inputs: &[&TensorRef]) -> TensorData {
        let x = &inputs[0].borrow().data;

        match x {
            TensorData::Scalar(_) => x.clone(),
            TensorData::Tensor(arr) => {
                let result = if let Some(axes) = &self.axes {
                    let axes_sorted: Vec<usize> = {
                        let mut a = axes.clone();
                        a.sort();
                        a
                    };

                    let mut reduced = arr.clone();
                    for &ax in axes_sorted.iter().rev() {
                        reduced = reduced.sum_axis(Axis(ax));
                        if self.keepdims {
                            reduced = reduced.insert_axis(Axis(ax));
                        }
                    }
                    reduced
                }
                else {
                    let sum = arr.sum();
                    if self.keepdims {
                        let shape = vec![1; arr.ndim()];
                        ArrayD::from_elem(IxDyn(&shape), sum)
                    } else {
                        return TensorData::Scalar(sum);
                    }
                };
                TensorData::Tensor(result)
            }
        }
    }

    fn backward(&self, output: &TensorRef, grad_output: &TensorData) -> Vec<TensorData> {
        let input = &output.borrow().parents[0];
        let input_shape = match &input.borrow().data {
            TensorData::Tensor(arr) => arr.shape().to_vec(),
            TensorData::Scalar(_) => vec![]
        };
        
        let grad = match grad_output {
            TensorData::Scalar(val) => {
                if input_shape.is_empty() {
                    TensorData::Scalar(*val)
                } else {
                    TensorData::Tensor(ArrayD::from_elem(IxDyn(&input_shape), *val))
                }
            },

            TensorData::Tensor(arr) => {
                if let Some(axes) = &self.axes {
                    let mut expanded = arr.clone();
                    if !self.keepdims {
                        for &ax in axes {
                            expanded = expanded.insert_axis(Axis(ax));
                        }
                    }

                    let broadcasted = expanded.broadcast(IxDyn(&input_shape)).expect("Broadcast failed in backward").to_owned();
                    TensorData::Tensor(broadcasted)
                } else {
                    let val = arr.sum();
                    TensorData::Tensor(ArrayD::from_elem(IxDyn(&input_shape), val))
                }
            }
        };

        vec![grad]
    }

    fn name(&self) -> &'static str { "Sum" }
}

impl Op for Mean {
    fn forward(&self, inputs: &[&TensorRef]) -> TensorData {
        let x = &inputs[0].borrow().data;

        match x {
            TensorData::Scalar(_) => x.clone(),
            TensorData::Tensor(arr) => {
                let result = if let Some(axes) = &self.axes {
                    let axes_sorted: Vec<usize> = {
                        let mut a = axes.clone();
                        a.sort();
                        a
                    };

                    let mut reduced = arr.clone();
                    for &ax in axes_sorted.iter().rev() {
                        reduced = reduced.mean_axis(Axis(ax)).expect("Error in forward mean");
                        if self.keepdims {
                            reduced = reduced.insert_axis(Axis(ax));
                        }
                    }
                    reduced
                }
                else {
                    let mean = arr.mean().expect("Error in forward mean");
                    if self.keepdims {
                        let shape = vec![1; arr.ndim()];
                        ArrayD::from_elem(IxDyn(&shape), mean)
                    } else {
                        return TensorData::Scalar(mean);
                    }
                };
                TensorData::Tensor(result)
            }
        }
    }

    fn backward(&self, output: &TensorRef, grad_output: &TensorData) -> Vec<TensorData> {
        let input = &output.borrow().parents[0];
        let input_shape = match &input.borrow().data {
            TensorData::Tensor(arr) => arr.shape().to_vec(),
            TensorData::Scalar(_) => vec![]
        };

        let total_count = if let Some(axes) = &self.axes {
            axes.iter().map(|&ax| input_shape[ax]).product::<usize>() as f32
        } else {
            input_shape.iter().product::<usize>() as f32
        };
        
        let grad = match grad_output {
            TensorData::Scalar(val) => {
                if input_shape.is_empty() {
                    TensorData::Scalar(*val / total_count)
                } else {
                    let scale = val / total_count;
                    TensorData::Tensor(ArrayD::from_elem(IxDyn(&input_shape), scale))
                }
            },

            TensorData::Tensor(arr) => {
                if let Some(axes) = &self.axes {
                    let mut expanded = arr.clone();
                    if !self.keepdims {
                        for &ax in axes {
                            expanded = expanded.insert_axis(Axis(ax));
                        }
                    }

                    let broadcasted = expanded
                        .broadcast(IxDyn(&input_shape))
                        .expect("Broadcast failed in backward")
                        .to_owned()
                        / total_count;
                    TensorData::Tensor(broadcasted)
                } else {
                    let val = arr.mean().unwrap();
                    TensorData::Tensor(ArrayD::from_elem(IxDyn(&input_shape), val))
                }
            }
        };

        vec![grad]
    }

    fn name(&self) -> &'static str { "Mean" }
}

fn apply_reduction_op(a: &TensorRef, op: Rc<dyn Op>) -> TensorRef {
    let data = op.forward(&[a]);
    let requires_grad = a.borrow().requires_grad;
    let result = Tensor::new(data, requires_grad);

    if requires_grad {
        result.borrow_mut().parents = vec![a.clone()];
        result.borrow_mut().grad_fn = Some(op);
    }

    result
}

pub fn sum(a: &TensorRef, axes: Option<Vec<usize>>, keepdim: bool) -> TensorRef {
    apply_reduction_op(a, Rc::new(Sum {axes: axes, keepdims: keepdim}))
}

pub fn mean(a: &TensorRef, axes: Option<Vec<usize>>, keepdim: bool) -> TensorRef {
    apply_reduction_op(a, Rc::new(Mean {axes: axes, keepdims: keepdim}))
}