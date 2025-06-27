mod tensor;
mod ops;

use tensor::Tensor;
use tensor::TensorOps;
use ops::add;
use ops::mul;

fn main() {
    let x = Tensor::new(5.0, true);
    let two = Tensor::new(2.0, true);
    let one = Tensor::new(1.0, true);
    let c = mul(&two, &x); // c = 2 * x
    let d = add(&c, &one);
    d.backward();

    println!("Grad x: {:?}", x.borrow().grad);
}