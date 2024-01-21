extern crate cmake;
use cmake::Config;
use std::env;
use std::path::PathBuf;

fn main() {
    let source = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("source");

    let dst = Config::new(&source)
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