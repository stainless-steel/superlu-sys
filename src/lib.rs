//! Bindings to [SuperLU].
//!
//! [superlu]: http://crd-legacy.lbl.gov/~xiaoye/SuperLU

#![allow(bad_style)]

extern crate libc;
extern crate openblas_src;

mod dsp_blas2;
mod dsp_blas3;
mod slu_ddefs;
mod slu_util;
mod superlu_enum_consts;
mod supermatrix;
mod sp_preorder;

pub use dsp_blas2::*;
pub use dsp_blas3::*;
pub use slu_ddefs::*;
pub use slu_util::*;
pub use sp_preorder::*;
pub use superlu_enum_consts::*;
pub use supermatrix::*;
