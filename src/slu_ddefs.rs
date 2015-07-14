use libc::{c_double, c_int};

use slu_util::*;
use supermatrix::*;

extern "C" {
    pub fn dgssv(options: *mut superlu_options_t, A: *mut SuperMatrix, perm_c: *mut c_int,
                 perm_r: *mut c_int, L: *mut SuperMatrix, U: *mut SuperMatrix, B: *mut SuperMatrix,
                 stat: *mut SuperLUStat_t, info: *mut c_int);

    pub fn dCreate_CompCol_Matrix(A: *mut SuperMatrix, m: c_int, n: c_int, nnz: c_int,
                                  nzval: *mut c_double, rowind: *mut c_int, colptr: *mut c_int,
                                  stype: Stype_t, dtype: Dtype_t, mtype: Mtype_t);

    pub fn dCreate_Dense_Matrix(X: *mut SuperMatrix, n: c_int, m: c_int, x: *mut c_double,
                                idx: c_int, stype: Stype_t, dtype: Dtype_t, mtype: Mtype_t);

    pub fn doubleMalloc(n: c_int) -> *mut c_double;
}
