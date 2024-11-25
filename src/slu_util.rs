use libc::{c_double, c_float, c_int, c_void};

use superlu_enum_consts::*;
use supermatrix::*;

pub type flops_t = c_float;

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

#[repr(C)]
pub struct GlobalLU_t {
    pub xsup: *mut c_int,
    pub supno: *mut c_int,
    pub lsub: *mut c_int,
    pub xlsub: *mut c_int,
    pub lusup: *mut c_double,
    pub xlusup: *mut c_int,
    pub ucol: *mut c_double,
    pub usub: *mut c_int,
    pub xusub: *mut c_int,
    pub nzlmax: c_int,
    pub nzumax: c_int,
    pub nzlumax: c_int,
    pub num_expansions: c_int,
}

extern "C" {
    pub fn Destroy_SuperMatrix_Store(A: *mut SuperMatrix);
    pub fn Destroy_CompCol_Matrix(A: *mut SuperMatrix);
    pub fn Destroy_CompRow_Matrix(A: *mut SuperMatrix);
    pub fn Destroy_SuperNode_Matrix(A: *mut SuperMatrix);
    pub fn Destroy_CompCol_Permuted(A: *mut SuperMatrix);
    pub fn Destroy_Dense_Matrix(A: *mut SuperMatrix);
    pub fn set_default_options(options: *mut superlu_options_t);
    pub fn intMalloc(n: c_int) -> *mut c_int;
    pub fn superlu_free(addr: *mut c_void);
    pub fn StatInit(stat: *mut SuperLUStat_t);
    pub fn StatFree(stat: *mut SuperLUStat_t);
    pub fn sp_ienv(ispec: c_int) -> c_int;
}

pub static SUPERLU_FREE: unsafe extern "C" fn(*mut c_void) = superlu_free;
