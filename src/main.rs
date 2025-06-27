mod tensor;
mod ops;

use tensor::Tensor;
use tensor::TensorOps;
use ops::add;
use ops::sub;
use ops::mul;
use ops::div;

fn main() {
    let x = Tensor::new(2.0, true);
    let ten = Tensor::new(10.0, false);
    let c = div(&ten, &x); // c = 10 / x
    c.backward();
    
    println!("Grad x: {:?}", x.borrow().grad); // Some(2.5)
}