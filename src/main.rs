mod tensor;
mod ops;

use tensor::Tensor;
use tensor::TensorOps;

use ndarray::arr0;
use ops::add;
use ops::sub;
use ops::mul;
use ops::div;

fn main() {
    let a = Tensor::new(arr0(2.0).into_dyn(), false);
    let b = Tensor::new(arr0(3.0).into_dyn(), false);
    let x = Tensor::new(arr0(10.0).into_dyn(), true);
    let c = add(&mul(&a, &x), &b);
    c.backward();

    println!("Grad a: {:?}", a.borrow().grad);
    println!("Grad b: {:?}", b.borrow().grad);
    println!("Grad x: {:?}", x.borrow().grad);
}