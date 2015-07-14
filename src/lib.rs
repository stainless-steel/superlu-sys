//! Bindings to [SuperLU][1].
//!
//! [1]: http://crd-legacy.lbl.gov/~xiaoye/SuperLU

#![allow(bad_style)]

extern crate blas_sys;
extern crate libc;

mod slu_ddefs;
mod slu_util;
mod superlu_enum_consts;
mod supermatrix;

pub use slu_ddefs::*;
pub use slu_util::*;
pub use superlu_enum_consts::*;
pub use supermatrix::*;
