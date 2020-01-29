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
pub enum LU_space_t {
    SYSTEM,
    USER,
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
