//! Bindings to [SuperLU][1].
//!
//! [1]: http://crd-legacy.lbl.gov/~xiaoye/SuperLU

#![allow(bad_style)]

extern crate blas_sys;
extern crate libc;

use libc::{c_double, c_float, c_int, c_void};

pub type flops_t = c_float;
pub type int_t = c_int;

#[derive(Clone, Copy)]
#[repr(C)]
pub enum Stype_t {
    SLU_NC,
    SLU_NCP,
    SLU_NR,
    SLU_SC,
    SLU_SCP,
    SLU_SR,
    SLU_DN,
    SLU_NR_loc,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum Dtype_t {
    SLU_S,
    SLU_D,
    SLU_C,
    SLU_Z,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum Mtype_t {
    SLU_GE,
    SLU_TRLU,
    SLU_TRUU,
    SLU_TRL,
    SLU_TRU,
    SLU_SYL,
    SLU_SYU,
    SLU_HEL,
    SLU_HEU,
}

#[allow(raw_pointer_derive)]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct SuperMatrix {
    pub Stype: Stype_t,
    pub Dtype: Dtype_t,
    pub Mtype: Mtype_t,
    pub nrow: int_t,
    pub ncol: int_t,
    pub Store: *mut c_void,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum yes_no_t {
    NO,
    YES,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum fact_t {
    DOFACT,
    SamePattern,
    SamePattern_SameRowPerm,
    FACTORED,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum rowperm_t {
    NOROWPERM,
    LargeDiag,
    MY_PERMR,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum colperm_t {
    NATURAL,
    MMD_ATA,
    MMD_AT_PLUS_A,
    COLAMD,
    METIS_AT_PLUS_A,
    PARMETIS,
    ZOLTAN,
    MY_PERMC,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum trans_t {
    NOTRANS,
    TRANS,
    CONJ,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum IterRefine_t {
    NOREFINE,
    SLU_SINGLE,
    SLU_DOUBLE,
    SLU_EXTRA,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum norm_t {
    ONE_NORM,
    TWO_NORM,
    INF_NORM,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum milu_t {
    SILU,
    SMILU_1,
    SMILU_2,
    SMILU_3,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct superlu_options_t {
    pub Fact: fact_t,
    pub Equil: yes_no_t,
    pub ColPerm: colperm_t,
    pub Trans: trans_t,
    pub IterRefine: IterRefine_t,
    pub DiagPivotThresh: c_double,
    pub SymmetricMode: yes_no_t,
    pub PivotGrowth: yes_no_t,
    pub ConditionNumber: yes_no_t,
    pub RowPerm: rowperm_t,
    pub ILU_DropRule: c_int,
    pub ILU_DropTol: c_double,
    pub ILU_FillFactor: c_double,
    pub ILU_Norm: norm_t,
    pub ILU_FillTol: c_double,
    pub ILU_MILU: milu_t,
    pub ILU_MILU_Dim: c_double,
    pub ParSymbFact: yes_no_t,
    pub ReplaceTinyPivot: yes_no_t,
    pub SolveInitialized: yes_no_t,
    pub RefineInitialized: yes_no_t,
    pub PrintStat: yes_no_t,
    pub nnzL: c_int,
    pub nnzU: c_int,
    pub num_lookaheads: c_int,
    pub lookahead_etree: yes_no_t,
    pub SymPattern: yes_no_t,
}

#[allow(raw_pointer_derive)]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct SuperLUStat_t {
    pub panel_histo: *mut c_int,
    pub utime: *mut c_double,
    pub ops: *mut flops_t,
    pub TinyPivots: c_int,
    pub RefineSteps: c_int,
    pub expansions: c_int,
}

// slu_ddefs.h
extern "C" {
    pub fn dCreate_CompCol_Matrix(A: *mut SuperMatrix, m: c_int, n: c_int, nnz: c_int,
                                  nzval: *mut c_double, rowind: *mut c_int, colptr: *mut c_int,
                                  stype: Stype_t, dtype: Dtype_t, mtype: Mtype_t);
    pub fn dCreate_Dense_Matrix(X: *mut SuperMatrix, n: c_int, m: c_int, x: *mut c_double,
                                idx: c_int, stype: Stype_t, dtype: Dtype_t, mtype: Mtype_t);

    pub fn doubleMalloc(n: c_int) -> *mut c_double;
}

// slu_util.h
extern "C" {
    pub fn Destroy_CompCol_Matrix(A: *mut SuperMatrix);
    pub fn Destroy_SuperMatrix_Store(A: *mut SuperMatrix);
    pub fn Destroy_SuperNode_Matrix(A: *mut SuperMatrix);

    pub fn StatFree(stat: *mut SuperLUStat_t);
    pub fn StatInit(stat: *mut SuperLUStat_t);

    pub fn intMalloc(n: c_int) -> *mut c_int;

    pub fn set_default_options(options: *mut superlu_options_t);

    pub fn superlu_free(addr: *mut c_void);

    pub fn dgssv(options: *mut superlu_options_t, A: *mut SuperMatrix, perm_c: *mut c_int,
                 perm_r: *mut c_int, L: *mut SuperMatrix, U: *mut SuperMatrix, B: *mut SuperMatrix,
                 stat: *mut SuperLUStat_t, info: *mut c_int);
}

pub static SUPERLU_FREE: unsafe extern fn(*mut c_void) = superlu_free;
