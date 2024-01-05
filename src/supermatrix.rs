use libc::{c_int, c_void};
use std::slice;

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

impl SuperMatrix {
    pub fn data_to_vec(&self) -> Option<Vec<f64>> {
        match self.Stype {
            Stype_t::SLU_DN => {}
            _ => return None,
        }

        match self.Dtype {
            Dtype_t::SLU_D => {}
            _ => return None,
        }

        unsafe {
            let store = self.Store as *const DNformat;
            if store.is_null() {
                return None;
            }

            let dn_format = &*store;
            let nnz = (self.nrow * self.ncol) as usize;
            match self.Dtype {
                Dtype_t::SLU_D => {
                    let values = slice::from_raw_parts(dn_format.nzval as *const f64, nnz);
                    Some(values.to_vec())
                }
                _ => None,
            }
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct NCformat {
    pub nnz: int_t,
    pub nzval: *mut c_void,
    pub rowind: *mut int_t,
    pub colptr: *mut int_t,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct DNformat {
    pub lda: int_t,
    pub nzval: *mut c_void,
}
