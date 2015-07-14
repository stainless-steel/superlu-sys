use libc::{c_char, c_double, c_int};

use slu_util::*;
use supermatrix::*;

extern "C" {
    pub fn dlsolve(ldm: c_int, ncol: c_int, M: *mut c_double, rhs: *mut c_double);

    pub fn dmatvec(ldm: c_int, nrow: c_int, ncol: c_int, M: *mut c_double, vec: *mut c_double,
                   Mxvec: *mut c_double);

    pub fn dusolve(ldm: c_int, ncol: c_int, M: *mut c_double, rhs: *mut c_double);

    pub fn sp_dgemv(trans: *mut c_char, alpha: c_double, A: *mut SuperMatrix, x: *mut c_double,
                    incx: c_int, beta: c_double, y: *mut c_double, incy: c_int) -> c_int;

    pub fn sp_dtrsv(uplo: *mut c_char, trans: *mut c_char, diag: *mut c_char, L: *mut SuperMatrix,
                    U: *mut SuperMatrix, x: *mut c_double, stat: *mut SuperLUStat_t,
                    info: *mut c_int) -> c_int;
}
