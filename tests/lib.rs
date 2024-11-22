extern crate libc;
extern crate superlu_sys as raw;

use std::mem;
use std::ptr::null_mut;
use libc::{c_char, c_double, c_int, c_void};
use raw::Dtype_t::SLU_D;
use raw::Mtype_t::SLU_GE;
use raw::Stype_t::{SLU_DN, SLU_NC};
use raw::*;
use raw::{dCreate_Dense_Matrix, doubleMalloc, SuperMatrix};
use std::slice::from_raw_parts_mut;
use raw::colperm_t::NATURAL;

// https://github.com/copies/superlu/blob/master/EXAMPLE/superlu.c
#[allow(non_snake_case)]
#[test]
fn test_valid() {
    use raw::colperm_t::*;
    use raw::Dtype_t::*;
    use raw::Mtype_t::*;
    use raw::Stype_t::*;

    unsafe {
        let (m, n, nnz) = (5, 5, 12);

        let a = doubleMalloc(nnz);
        assert!(!a.is_null());
        {
            let (s, u, p, e, r, l) = (19.0, 21.0, 16.0, 5.0, 18.0, 12.0);
            let a = from_raw_parts_mut(a, nnz as usize);
            a[0] = s;
            a[1] = l;
            a[2] = l;
            a[3] = u;
            a[4] = l;
            a[5] = l;
            a[6] = u;
            a[7] = p;
            a[8] = u;
            a[9] = e;
            a[10] = u;
            a[11] = r;
        }

        let asub = intMalloc(nnz);
        assert!(!asub.is_null());
        {
            let asub = from_raw_parts_mut(asub, nnz as usize);
            asub[0] = 0;
            asub[1] = 1;
            asub[2] = 4;
            asub[3] = 1;
            asub[4] = 2;
            asub[5] = 4;
            asub[6] = 0;
            asub[7] = 2;
            asub[8] = 0;
            asub[9] = 3;
            asub[10] = 3;
            asub[11] = 4;
        }

        let xa = intMalloc(n + 1);
        assert!(!xa.is_null());
        {
            let xa = from_raw_parts_mut(xa, (n + 1) as usize);
            xa[0] = 0;
            xa[1] = 3;
            xa[2] = 6;
            xa[3] = 8;
            xa[4] = 10;
            xa[5] = 12;
        }

        let mut A: SuperMatrix = std::mem::zeroed();

        dCreate_CompCol_Matrix(&mut A, m, n, nnz, a, asub, xa, SLU_NC, SLU_D, SLU_GE);

        let nrhs = 1;
        let rhs = doubleMalloc(m * nrhs);
        assert!(!rhs.is_null());
        {
            let rhs = from_raw_parts_mut(rhs, (m * nrhs) as usize);
            for i in 0..((m * nrhs) as usize) {
                rhs[i] = 1.0;
            }
        }

        let mut B: SuperMatrix = std::mem::zeroed();
        dCreate_Dense_Matrix(&mut B, m, nrhs, rhs, m, SLU_DN, SLU_D, SLU_GE);

        let perm_r = intMalloc(m);
        assert!(!perm_r.is_null());

        let perm_c = intMalloc(n);
        assert!(!perm_c.is_null());

        let mut options: superlu_options_t = std::mem::zeroed();
        set_default_options(&mut options);
        options.ColPerm = NATURAL;

        let mut stat: SuperLUStat_t = std::mem::zeroed();
        StatInit(&mut stat);

        let mut L: SuperMatrix = std::mem::zeroed();
        let mut U: SuperMatrix = std::mem::zeroed();

        let mut info = 0;
        dgssv(
            &mut options,
            &mut A,
            perm_c,
            perm_r,
            &mut L,
            &mut U,
            &mut B,
            &mut stat,
            &mut info,
        );

        SUPERLU_FREE(rhs as *mut _);
        SUPERLU_FREE(perm_r as *mut _);
        SUPERLU_FREE(perm_c as *mut _);
        Destroy_CompCol_Matrix(&mut A);
        Destroy_SuperMatrix_Store(&mut B);
        Destroy_SuperNode_Matrix(&mut L);
        Destroy_CompCol_Matrix(&mut U);
        StatFree(&mut stat);
    }
}

#[allow(non_snake_case)]
#[test]
fn test_singular() {
    use raw::colperm_t::*;
    use raw::Dtype_t::*;
    use raw::Mtype_t::*;
    use raw::Stype_t::*;

    unsafe {
        let (m, n, nnz) = (5, 5, 12);

        let a = doubleMalloc(nnz);
        assert!(!a.is_null());
        {
            let (s, u, p, e, r, l) = (0.0, 0.0, 16.0, 0.0, 0.0, 0.0);
            let a = from_raw_parts_mut(a, nnz as usize);
            a[0] = s;
            a[1] = l;
            a[2] = l;
            a[3] = u;
            a[4] = l;
            a[5] = l;
            a[6] = u;
            a[7] = p;
            a[8] = u;
            a[9] = e;
            a[10] = u;
            a[11] = r;
        }

        let asub = intMalloc(nnz);
        assert!(!asub.is_null());
        {
            let asub = from_raw_parts_mut(asub, nnz as usize);
            asub[0] = 0;
            asub[1] = 1;
            asub[2] = 4;
            asub[3] = 1;
            asub[4] = 2;
            asub[5] = 4;
            asub[6] = 0;
            asub[7] = 2;
            asub[8] = 0;
            asub[9] = 3;
            asub[10] = 3;
            asub[11] = 4;
        }

        let xa = intMalloc(n + 1);
        assert!(!xa.is_null());
        {
            let xa = from_raw_parts_mut(xa, (n + 1) as usize);
            xa[0] = 0;
            xa[1] = 3;
            xa[2] = 6;
            xa[3] = 8;
            xa[4] = 10;
            xa[5] = 12;
        }

        let mut A: SuperMatrix = std::mem::zeroed();

        dCreate_CompCol_Matrix(&mut A, m, n, nnz, a, asub, xa, SLU_NC, SLU_D, SLU_GE);

        let nrhs = 1;
        let rhs = doubleMalloc(m * nrhs);
        assert!(!rhs.is_null());
        {
            let rhs = from_raw_parts_mut(rhs, (m * nrhs) as usize);
            for i in 0..((m * nrhs) as usize) {
                rhs[i] = 1.0;
            }
        }

        let mut B: SuperMatrix = std::mem::zeroed();
        dCreate_Dense_Matrix(&mut B, m, nrhs, rhs, m, SLU_DN, SLU_D, SLU_GE);

        let perm_r = intMalloc(m);
        assert!(!perm_r.is_null());

        let perm_c = intMalloc(n);
        assert!(!perm_c.is_null());

        let mut options: superlu_options_t = std::mem::zeroed();
        set_default_options(&mut options);
        options.ColPerm = NATURAL;

        let mut stat: SuperLUStat_t = std::mem::zeroed();
        StatInit(&mut stat);

        let mut L: SuperMatrix = std::mem::zeroed();
        let mut U: SuperMatrix = std::mem::zeroed();

        let mut info = 0;
        dgssv(
            &mut options,
            &mut A,
            perm_c,
            perm_r,
            &mut L,
            &mut U,
            &mut B,
            &mut stat,
            &mut info,
        );
        assert_eq!(info, 1);
        SUPERLU_FREE(rhs as *mut _);
        SUPERLU_FREE(perm_r as *mut _);
        SUPERLU_FREE(perm_c as *mut _);
        Destroy_CompCol_Matrix(&mut A);
        Destroy_SuperMatrix_Store(&mut B);
        Destroy_SuperNode_Matrix(&mut L);
        Destroy_CompCol_Matrix(&mut U);
        StatFree(&mut stat);
    }
}

#[test]
fn test_read_write_super_matrix_f64() {
    let m = 3;
    let n = 2;
    let v = unsafe {
        let rhs = doubleMalloc(m * n);
        assert!(!rhs.is_null());
        {
            let rhs = from_raw_parts_mut(rhs, (m * n) as usize);
            for i in 0..((m * n) as usize) {
                rhs[i] = i as f64;
            }
        }

        let mut a: SuperMatrix = std::mem::zeroed();
        dCreate_Dense_Matrix(&mut a, m, n, rhs, m, SLU_DN, SLU_D, SLU_GE);
        a.data_to_vec().unwrap()
    };
    for i in 0..v.len() as i32 {
        assert_eq!(v[i as usize], i.into());
    }
}

#[allow(non_snake_case)]
#[test]
fn test_sp_dgemv() {
    use raw::Dtype_t::*;
    use raw::Mtype_t::*;
    use raw::Stype_t::*;
    use std::ffi::CString;

    unsafe {
        let m = 3;
        let n = 3;
        let nnz = 4;

        let a = doubleMalloc(nnz);
        assert!(!a.is_null());
        {
            let a = from_raw_parts_mut(a, nnz as usize);
            a[0] = 1.0;
            a[1] = 2.0;
            a[2] = 3.0;
            a[3] = 4.0;
        }

        let asub = intMalloc(nnz);
        assert!(!asub.is_null());
        {
            let asub = from_raw_parts_mut(asub, nnz as usize);
            asub[0] = 0;
            asub[1] = 1;
            asub[2] = 2;
            asub[3] = 0;
        }

        let xa = intMalloc(n + 1);
        assert!(!xa.is_null());
        {
            let xa = from_raw_parts_mut(xa, (n + 1) as usize);
            xa[0] = 0;
            xa[1] = 1;
            xa[2] = 2;
            xa[3] = 4;
        }

        let mut A: SuperMatrix = std::mem::zeroed();
        dCreate_CompCol_Matrix(&mut A, m, n, nnz, a, asub, xa, SLU_NC, SLU_D, SLU_GE);

        let x = doubleMalloc(n);
        assert!(!x.is_null());
        {
            let x = from_raw_parts_mut(x, n as usize);
            x[0] = 1.0;
            x[1] = 1.0;
            x[2] = 1.0;
        }

        let raw_y = doubleMalloc(m);
        assert!(!raw_y.is_null());

        let y_slice = from_raw_parts_mut(raw_y, m as usize);
        for i in 0..m as usize {
            y_slice[i] = 0.0;
        }

        let alpha = 1.0;
        let beta = 0.0;

        let trans = CString::new("N").unwrap();

        let incx = 1;
        let incy = 1;

        let info = sp_dgemv(
            trans.as_ptr() as *mut c_char,
            alpha,
            &mut A,
            x,
            incx,
            beta,
            raw_y,
            incy,
        );
        assert_eq!(info, 0);

        let y = from_raw_parts_mut(raw_y, m as usize);
        assert_eq!(y[0], 5.0);
        assert_eq!(y[1], 2.0);
        assert_eq!(y[2], 3.0);

        SUPERLU_FREE(a as *mut _);
        SUPERLU_FREE(asub as *mut _);
        SUPERLU_FREE(xa as *mut _);
        SUPERLU_FREE(x as *mut _);
        SUPERLU_FREE(raw_y as *mut _);
    }
}

#[test]
fn test_dgstrf() {
    unsafe {
        // Define the dimensions of the matrix
        let m: c_int = 5;  // Number of rows
        let n: c_int = 5;  // Number of columns
        let nnz: c_int = 12;  // Number of non-zero entries

        // Define the non-zero entries of the matrix A
        let a: [c_double; 12] = [
            19.0, 21.0, 21.0, 16.0, 5.0, 18.0,
            12.0, 12.0, 17.0, 5.0, 18.0, 5.0,
        ];

        // Row indices of the non-zero entries
        let asub: [c_int; 12] = [
            0, 1, 2, 0, 2, 3,
            1, 2, 4, 2, 3, 4,
        ];

        // Column pointers
        let xa: [c_int; 6] = [0, 3, 5, 8, 10, 12];

        // Allocate and initialize the SuperMatrix A
        let mut A = mem::zeroed();
        dCreate_CompCol_Matrix(
            &mut A,
            m,
            n,
            nnz,
            a.as_ptr() as *mut c_double,
            asub.as_ptr() as *mut c_int,
            xa.as_ptr() as *mut c_int,
            Stype_t::SLU_NC,  // Compressed Column format
            Dtype_t::SLU_D,   // Double precision
            Mtype_t::SLU_GE,  // General matrix
        );

        // Prepare permutation vectors
        let mut perm_r = vec![0 as c_int; m as usize];
        let mut perm_c = vec![0 as c_int; n as usize];

        // Initialize options with default values
        let mut options= mem::zeroed();
        set_default_options(&mut options);
        options.ColPerm = colperm_t::COLAMD;  // Use COLAMD for column permutation

        // Initialize SuperLUStat_t
        let mut stat= mem::zeroed();
        StatInit(&mut stat);

        // Prepare L and U matrices
        let mut L= mem::zeroed();
        let mut U= mem::zeroed();

        // Prepare GlobalLU_t structure
        let mut Glu = GlobalLU_t {
            xsup: null_mut(),
            supno: null_mut(),
            lsub: null_mut(),
            xlsub: null_mut(),
            lusup: null_mut(),
            xlusup: null_mut(),
            ucol: null_mut(),
            usub: null_mut(),
            xusub: null_mut(),
            nzlmax: 0,
            nzumax: 0,
            nzlumax: 0,
            num_expansions: 0,
        };

        // Compute the elimination tree
        let mut etree = vec![0 as c_int; n as usize];

        // Allocate a SuperMatrix for AC
        let mut AC= mem::zeroed();

        // Perform the symbolic factorization and compute the permuted matrix AC
        sp_preorder(
            &mut options,
            &mut A,
            perm_c.as_mut_ptr(),
            etree.as_mut_ptr(),
            &mut AC,
        );

        // Prepare other parameters
        let relax: c_int = sp_ienv(2);  // Number of columns in a relaxed supernode
        let panel_size: c_int = sp_ienv(1);  // Panel size
        let work: *mut c_void = null_mut();  // Workspace (not used here)
        let lwork: c_int = 0;  // Specifies the size of work array
        let mut info: c_int = 0;

        // Call dgstrf to perform LU factorization on the permuted matrix AC
        dgstrf(
            &mut options,
            &mut AC,
            relax,
            panel_size,
            etree.as_mut_ptr(),
            work,
            lwork,
            perm_c.as_mut_ptr(),
            perm_r.as_mut_ptr(),
            &mut L,
            &mut U,
            &mut Glu,
            &mut stat,
            &mut info,
        );

        // Check that factorization was successful
        assert_eq!(info, 0, "dgstrf failed with info = {}", info);

        // Optionally, you can inspect the contents of L and U
        // For example, print the number of non-zeros in L and U
        let Lstore = L.Store as *mut SCformat;
        let Ustore = U.Store as *mut NCformat;

        println!("Number of non-zeros in L = {}", (*Lstore).nnz);
        println!("Number of non-zeros in U = {}", (*Ustore).nnz);

        // Clean up and free allocated resources
        Destroy_SuperMatrix_Store(&mut A);
        Destroy_SuperNode_Matrix(&mut L);
        Destroy_CompCol_Matrix(&mut U);
        Destroy_CompCol_Permuted(&mut AC);  // Clean up AC
        StatFree(&mut stat);
    }
}
