extern crate cmake;
use cmake::Config;
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let source = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("source");

    let source_copy = out_dir.join("source");
    if source_copy.exists() {
        fs::remove_dir_all(&source_copy).unwrap();
    }
    fs::create_dir_all(&source_copy).unwrap();
    copy_dir_all(&source, &source_copy).unwrap();

    let dst = Config::new(&source_copy)
        .define("CMAKE_BUILD_TYPE", "Release")
        .cflag("-O3 -DNDEBUG -DPRNTlevel=0 -fPIC -w")
        .build();

    let lib_path = dst.join("lib");

    if !lib_path.join("libsuperlu.a").exists() {
        panic!("Could not find native static library `superlu`");
    }

    println!("cargo:rustc-link-lib=static=superlu");
    println!("cargo:rustc-link-search=native={}", lib_path.display());
}

fn copy_dir_all(src: &PathBuf, dst: &PathBuf) -> std::io::Result<()> {
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        if ty.is_dir() {
            let new_dst = dst.join(entry.file_name());
            fs::create_dir_all(&new_dst)?;
            copy_dir_all(&entry.path(), &new_dst)?;
        } else {
            fs::copy(entry.path(), dst.join(entry.file_name()))?;
        }
    }
    Ok(())
}
