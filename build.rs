use std::{env, process};
use std::path::PathBuf;

macro_rules! cmd(
    ($name:expr) => (process::Command::new($name));
);

macro_rules! get(
    ($name:expr) => (env::var($name).unwrap_or(String::new()));
);

macro_rules! run(
    ($command:expr) => (
        assert!($command.stdout(process::Stdio::inherit())
                        .stderr(process::Stdio::inherit())
                        .status().unwrap().success());
    );
);

fn main() {
    let source = PathBuf::from(&get!("CARGO_MANIFEST_DIR")).join("superlu");
    let output = PathBuf::from(&get!("OUT_DIR"));

    run!(cmd!("make").current_dir(&source)
                     .arg("superlulib")
                     .arg("CFLAGS=-DPRNTlevel=0 -O3 -Wno-logical-op-parentheses")
                     .arg(&format!("SuperLUroot={}", source.display()))
                     .arg(&format!("SUPERLULIB={}", output.join("libsuperlu.a").display())));

    println!("cargo:rustc-link-lib=static=superlu");
    println!("cargo:rustc-link-search={}", output.display());
}
