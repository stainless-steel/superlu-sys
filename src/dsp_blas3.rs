use libc::{c_char, c_double, c_int};

use supermatrix::*;

extern "C" {
    pub fn sp_dgemm(transa: *mut c_char, transb: *mut c_char, m: c_int, n: c_int, k: c_int,
                    alpha: c_double, A: *mut SuperMatrix, b: *mut c_double, ldb: c_int,
                    beta: c_double, c: *mut c_double, ldc: c_int) -> c_int;
}
