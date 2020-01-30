extern crate libc;
extern crate superlu_sys as raw;

// https://github.com/copies/superlu/blob/master/EXAMPLE/superlu.c
#[allow(non_snake_case)]
#[test]
fn test_dgstrf_dgstrs() {
    use libc::c_int;
    use std::mem::MaybeUninit;
    use std::slice::from_raw_parts_mut;

    use raw::colperm_t::*;
    use raw::fact_t::*;
    use raw::trans_t::*;
    use raw::Dtype_t::*;
    use raw::Mtype_t::*;
    use raw::Stype_t::*;
    use raw::*;

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

        let mut A = MaybeUninit::<SuperMatrix>::uninit();
        dCreate_CompCol_Matrix(
            A.as_mut_ptr(),
            m,
            n,
            nnz,
            a,
            asub,
            xa,
            SLU_NC,
            SLU_D,
            SLU_GE,
        );
        let mut A = A.assume_init();

        let nrhs = 1;
        let rhs = doubleMalloc(m * nrhs);
        assert!(!rhs.is_null());
        {
            let rhs = from_raw_parts_mut(rhs, (m * nrhs) as usize);
            for i in 0..((m * nrhs) as usize) {
                rhs[i] = 1.0;
            }
        }

        let mut B = MaybeUninit::<SuperMatrix>::uninit();
        dCreate_Dense_Matrix(B.as_mut_ptr(), m, nrhs, rhs, m, SLU_DN, SLU_D, SLU_GE);
        let mut B = B.assume_init();

        let perm_r = intMalloc(m);
        assert!(!perm_r.is_null());

        let perm_c = intMalloc(n);
        assert!(!perm_c.is_null());

        let mut options = MaybeUninit::<superlu_options_t>::uninit();
        set_default_options(options.as_mut_ptr());
        let mut options = options.assume_init();
        options.ColPerm = MMD_ATA;

        let mut stat = MaybeUninit::<SuperLUStat_t>::uninit();
        StatInit(stat.as_mut_ptr());
        let mut stat = stat.assume_init();

        let mut L = MaybeUninit::<SuperMatrix>::uninit();
        let mut U = MaybeUninit::<SuperMatrix>::uninit();

        let mut info = 0;
        poor_mans_dgssv(
            &mut options,
            &mut A,
            perm_c,
            perm_r,
            L.as_mut_ptr(),
            U.as_mut_ptr(),
            &mut B,
            &mut stat,
            &mut info,
        );
        let mut L = L.assume_init();
        let mut U = U.assume_init();

        SUPERLU_FREE(rhs as *mut _);
        SUPERLU_FREE(perm_r as *mut _);
        SUPERLU_FREE(perm_c as *mut _);
        Destroy_CompCol_Matrix(&mut A);
        Destroy_SuperMatrix_Store(&mut B);
        Destroy_SuperNode_Matrix(&mut L);
        Destroy_CompCol_Matrix(&mut U);
        StatFree(&mut stat);
    }

    // implementation of dgssv which leaves out some timing info,
    // and which doesn't support NR format
    //
    // for testing dgstrf and dgstrs (as well as get_perm_c and sp_preorder)
    unsafe fn poor_mans_dgssv(
        options: &mut superlu_options_t,
        A: &mut SuperMatrix,
        perm_c: *mut c_int,
        perm_r: *mut c_int,
        L: *mut SuperMatrix,
        U: *mut SuperMatrix,
        B: &mut SuperMatrix,
        stat: &mut SuperLUStat_t,
        info: &mut c_int,
    ) {
        *info = 0;

        assert_eq!(options.Fact as i32, DOFACT as i32);
        assert_eq!(A.nrow, A.ncol);
        assert!(A.nrow >= 0);
        assert_eq!(A.Stype as i32, SLU_NC as i32);
        assert_eq!(A.Dtype as i32, SLU_D as i32);
        assert_eq!(A.Mtype as i32, SLU_GE as i32);
        assert!(B.ncol >= 0);
        // assert!((*(B.Store as *mut DNFormat)).lda >= std::cmp::max(0, A.nrow));
        assert_eq!(B.Stype as i32, SLU_DN as i32);
        assert_eq!(B.Dtype as i32, SLU_D as i32);
        assert_eq!(B.Mtype as i32, SLU_GE as i32);

        let permc_spec = options.ColPerm;
        if permc_spec as i32 != MY_PERMC as i32 {
            get_perm_c(permc_spec as _, A, perm_c);
        }

        let etree = intMalloc(A.ncol);

        let mut AC = MaybeUninit::<SuperMatrix>::uninit();
        sp_preorder(options, A, perm_c, etree, AC.as_mut_ptr());

        let mut Glu = MaybeUninit::<GlobalLU_t>::uninit();
        let panel_size = sp_ienv(1);
        let relax = sp_ienv(2);
        let work = std::ptr::null_mut();
        let lwork = 0;
        dgstrf(
            options,
            AC.as_mut_ptr(),
            relax,
            panel_size,
            etree,
            work,
            lwork,
            perm_c,
            perm_r,
            L,
            U,
            Glu.as_mut_ptr(),
            stat,
            info,
        );
        let mut AC = AC.assume_init();
        let _ = Glu.assume_init();

        if *info == 0 {
            dgstrs(NOTRANS, L, U, perm_c, perm_r, B, stat, info);
        }

        SUPERLU_FREE(etree as *mut _);
        Destroy_CompCol_Permuted(&mut AC);

        assert_eq!(*info, 0);
    }
}
