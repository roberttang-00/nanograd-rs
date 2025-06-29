mod tensor;
mod ops;

use tensor::Tensor;
use tensor::TensorOps;

use ndarray::Array;
use ops::*;

fn main() {
    let a = Tensor::new(2.0, true);
    let b = Tensor::new(-3.0, true);
    let x = Tensor::new(Array::range(0.5, 2.0, 0.5).into_dyn(), true);
    let c = &add(&mul(&a, &x), &b);

    let target = Tensor::new(Array::from_vec(vec![-2.0, -1.0, 0.0]).into_dyn(), true);
    println!("{}", c.borrow());

    let error = &sub(&target, &c);
    error.backward();

    println!("Grad a: {:?}", a.borrow().grad);
    println!("Grad b: {:?}", b.borrow().grad);
    println!("Grad x: {:?}", x.borrow().grad);
}