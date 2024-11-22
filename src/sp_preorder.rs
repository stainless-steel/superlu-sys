use std::ffi::c_int;
use ::{superlu_options_t, SuperMatrix};

extern "C" {
    pub fn sp_preorder(
        options: *mut superlu_options_t,
        A: *mut SuperMatrix,
        perm_c: *mut c_int,
        etree: *mut c_int,
        AC: *mut SuperMatrix,
    );
}