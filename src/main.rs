mod tensor;
mod ops;

use tensor::Tensor;
use tensor::TensorOps;

use ndarray::arr0;
use ops::*;

fn main() {
    let a = Tensor::new(2.0, false);
    let b = Tensor::new(3.0, false);
    let x = Tensor::new(10.0, true);
    let c = add(&div(&a, &x), &b);
    c.backward();

    println!("Grad a: {:?}", a.borrow().grad);
    println!("Grad b: {:?}", b.borrow().grad);
    println!("Grad x: {:?}", x.borrow().grad);
}