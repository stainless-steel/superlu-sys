use libc::{c_double, c_int, c_void};

use slu_util::*;
use supermatrix::*;
use trans_t;

extern "C" {
    pub fn dgssv(
        options: *mut superlu_options_t,
        A: *mut SuperMatrix,
        perm_c: *mut c_int,
        perm_r: *mut c_int,
        L: *mut SuperMatrix,
        U: *mut SuperMatrix,
        B: *mut SuperMatrix,
        stat: *mut SuperLUStat_t,
        info: *mut c_int,
    );

    pub fn dgstrf(
        options: *mut superlu_options_t,
        A: *mut SuperMatrix,
        relax: c_int,
        panel_size: c_int,
        etree: *mut c_int,
        work: *mut c_void,
        lwork: c_int,
        perm_c: *mut c_int,
        perm_r: *mut c_int,
        L: *mut SuperMatrix,
        U: *mut SuperMatrix,
        Glu: *mut GlobalLU_t,
        stat: *mut SuperLUStat_t,
        info: *mut c_int,
    );

    pub fn dgstrs(
        trans: trans_t,
        L: *mut SuperMatrix,
        U: *mut SuperMatrix,
        perm_c: *mut c_int,
        perm_r: *mut c_int,
        B: *mut SuperMatrix,
        stat: *mut SuperLUStat_t,
        info: *mut c_int,
    );

    pub fn dCreate_CompCol_Matrix(
        A: *mut SuperMatrix,
        m: c_int,
        n: c_int,
        nnz: c_int,
        nzval: *mut c_double,
        rowind: *mut c_int,
        colptr: *mut c_int,
        stype: Stype_t,
        dtype: Dtype_t,
        mtype: Mtype_t,
    );

    pub fn dCreate_Dense_Matrix(
        X: *mut SuperMatrix,
        n: c_int,
        m: c_int,
        x: *mut c_double,
        idx: c_int,
        stype: Stype_t,
        dtype: Dtype_t,
        mtype: Mtype_t,
    );

    pub fn doubleMalloc(n: c_int) -> *mut c_double;
}
