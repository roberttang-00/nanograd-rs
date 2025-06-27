use nanograd_rs::tensor::{Tensor, TensorOps};
use nanograd_rs::ops::{add, sub, mul, div};

#[test]
fn test_basic_forward_pass() {
    let x = Tensor::new(2.0, true);
    let y = Tensor::new(3.0, true);
    let result = add(&x, &y);
    
    assert_eq!(result.borrow().data, 5.0);
    assert_eq!(result.borrow().requires_grad, true);
}

#[test]
fn test_division_gradient() {
    let x = Tensor::new(2.0, true);
    let ten = Tensor::new(10.0, false);
    let result = div(&ten, &x); // 10 / x
    
    result.backward();
    
    // d/dx(10/x) = -10/x^2 = -10/4 = -2.5
    assert_eq!(result.borrow().data, 5.0);
    assert_eq!(x.borrow().grad, Some(-2.5));
}

#[test]
fn test_chain_rule() {
    // f(x) = (x + 3) * (x - 1)
    // f'(x) = 2x + 2
    // At x = 2: f'(2) = 6
    let x = Tensor::new(2.0, true);
    let three = Tensor::new(3.0, false);
    let one = Tensor::new(1.0, false);
    
    let x_plus_3 = add(&x, &three);
    let x_minus_1 = sub(&x, &one);
    let result = mul(&x_plus_3, &x_minus_1);
    
    result.backward();
    
    assert_eq!(result.borrow().data, 5.0);
    assert_eq!(x.borrow().grad, Some(6.0));
}

#[test]
fn test_multi_variable_gradients() {
    // f(x,y) = x * y + x
    // df/dx = y + 1 = 3 + 1 = 4
    // df/dy = x = 2
    let x = Tensor::new(2.0, true);
    let y = Tensor::new(3.0, true);
    
    let xy = mul(&x, &y);
    let result = add(&xy, &x);
    
    result.backward();
    
    assert_eq!(result.borrow().data, 8.0); // 2*3 + 2 = 8
    assert_eq!(x.borrow().grad, Some(4.0));
    assert_eq!(y.borrow().grad, Some(2.0));
}

#[test]
fn test_complex_expression() {
    // f(x) = (x * 2 + 1) / (x + 3)
    // Using quotient rule: f'(x) = [2(x+3) - (2x+1)] / (x+3)^2 = 5 / (x+3)^2
    // At x = 1: f'(1) = 5 / 16 = 0.3125
    let x = Tensor::new(1.0, true);
    let two = Tensor::new(2.0, false);
    let one = Tensor::new(1.0, false);
    let three = Tensor::new(3.0, false);
    
    let x_times_2 = mul(&x, &two);
    let numerator = add(&x_times_2, &one);
    let denominator = add(&x, &three);
    let result = div(&numerator, &denominator);
    
    result.backward();
    
    assert_eq!(result.borrow().data, 0.75); // (2+1)/(1+3) = 3/4 = 0.75
    assert!((x.borrow().grad.unwrap() - 0.3125).abs() < 1e-6);
}

#[test]
fn test_gradient_accumulation() {
    // Test that gradients accumulate when a variable is used multiple times
    // f(x) = x + x = 2x
    // f'(x) = 2
    let x = Tensor::new(5.0, true);
    let result = add(&x, &x);
    
    result.backward();
    
    assert_eq!(result.borrow().data, 10.0);
    assert_eq!(x.borrow().grad, Some(2.0));
}

#[test]
fn test_no_grad_propagation() {
    // Test that gradients don't propagate through non-grad tensors
    let x = Tensor::new(2.0, true);
    let y = Tensor::new(3.0, false); // requires_grad = false
    let result = mul(&x, &y);
    
    result.backward();
    
    assert_eq!(result.borrow().data, 6.0);
    assert_eq!(x.borrow().grad, Some(3.0));
    assert_eq!(y.borrow().grad, None);
}

#[test]
fn test_deep_computation_graph() {
    // Test deep nesting: ((x + 1) * 2 - 1) + x
    // Simplifies to: 2x + 2 - 1 + x = 3x + 1
    // f'(x) = 3
    let x = Tensor::new(2.0, true);
    let one = Tensor::new(1.0, false);
    let two = Tensor::new(2.0, false);
    
    let step1 = add(&x, &one);
    let step2 = mul(&step1, &two);
    let step3 = sub(&step2, &one);
    let result = add(&step3, &x);
    
    result.backward();
    
    assert_eq!(result.borrow().data, 7.0); // 3*2 + 1 = 7
    assert_eq!(x.borrow().grad, Some(3.0));
}

#[test]
#[should_panic(expected = "division by zero")]
fn test_division_by_zero() {
    let x = Tensor::new(5.0, true);
    let zero = Tensor::new(0.0, false);
    let _result = div(&x, &zero);
}

#[test]
fn test_performance_large_graph() {
    let start = std::time::Instant::now();
    
    let x = Tensor::new(1.0, true);
    let mut result = x.clone();
    
    for i in 1..100 {
        let val = Tensor::new(i as f32, false);
        result = add(&result, &val);
        result = mul(&result, &Tensor::new(0.99, false));
    }
    
    result.backward();
    
    let duration = start.elapsed();
    println!("Large graph computation took: {:?}", duration);
    
    assert!(x.borrow().grad.is_some());
}

fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
    (a - b).abs() < epsilon
}

#[test]
fn test_floating_point_precision() {
    let x = Tensor::new(0.1, true);
    let y = Tensor::new(0.2, true);
    let z = Tensor::new(0.3, false);
    
    let sum = add(&x, &y);
    let result = sub(&sum, &z);
    
    result.backward();
    
    assert!(approx_eq(result.borrow().data, 0.0, 1e-6));
    assert_eq!(x.borrow().grad, Some(1.0));
    assert_eq!(y.borrow().grad, Some(1.0));
}