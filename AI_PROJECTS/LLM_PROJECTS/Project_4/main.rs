use std::time::Instant;

const A: u64 = 1664525;
const C: u64 = 1013904223;
const MOD: u64 = 1u64 << 32;

#[inline(always)]
fn lcg_next(state: &mut u64) -> u64 {
    // (a * state + c) mod 2^32
    *state = (A.wrapping_mul(*state).wrapping_add(C)) & (MOD - 1);
    *state
}

fn max_subarray_sum(n: usize, seed: u64, min_val: i64, max_val: i64) -> i64 {
    let range = (max_val - min_val + 1) as u64;
    let mut gen_state = seed;
    let mut best = i64::MIN;
    let mut cur = 0i64;

    for _ in 0..n {
        let v = lcg_next(&mut gen_state);
        let x = (v % range) as i64 + min_val;
        cur = if cur > 0 { cur + x } else { x };
        if cur > best {
            best = cur;
        }
    }
    best
}

fn total_max_subarray_sum(n: usize, initial_seed: u64, min_val: i64, max_val: i64) -> i64 {
    let mut total = 0i64;
    let mut lcg_state = initial_seed;
    for _ in 0..20 {
        let seed = lcg_next(&mut lcg_state);
        total += max_subarray_sum(n, seed, min_val, max_val);
    }
    total
}

fn main() {
    // Parameters
    let n: usize = 10_000;
    let initial_seed: u64 = 42;
    let min_val: i64 = -10;
    let max_val: i64 = 10;

    let start = Instant::now();
    let result = total_max_subarray_sum(n, initial_seed, min_val, max_val);
    let duration = start.elapsed();

    println!("Total Maximum Subarray Sum (20 runs): {}", result);
    println!(
        "Execution Time: {:.6} seconds",
        duration.as_secs_f64()
    );
}