use std::{env, fs, io, process};
use std::path::{Path, PathBuf};

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

macro_rules! ok(
    ($result:expr) => (
        match $result {
            Ok(ok) => ok,
            Err(error) => panic!("`{}` failed with `{}`", stringify!($result), error),
        }
    );
);

#[allow(unused_must_use)]
fn main() {
    let source = PathBuf::from(&get!("CARGO_MANIFEST_DIR")).join("source");
    let output = PathBuf::from(&get!("OUT_DIR"));

    let include = output.join("include");
    fs::create_dir(&include);
    ok!(copy(&source.join("SRC"), &include, "h"));

    let lib = output.join("lib");
    fs::create_dir(&lib);
    run!(cmd!("make").current_dir(&source.join("SRC"))
                     .arg("NOOPTS=-O0 -fPIC -w")
                     .arg("CFLAGS=-O3 -DNDEBUG -DPRNTlevel=0 -fPIC -w")
                     .arg("DZAUX=")
                     .arg("SCAUX=")
                     .arg(&format!("SuperLUroot={}", source.display()))
                     .arg(&format!("SUPERLULIB={}", lib.join("libsuperlu.a").display())));

    println!("cargo:root={}", output.display());
    println!("cargo:rustc-link-lib=static=superlu");
    println!("cargo:rustc-link-search={}", lib.display());
}

fn copy(source: &Path, destination: &Path, extension: &str) -> io::Result<()> {
    for entry in try!(fs::read_dir(source)) {
        let path = try!(entry).path();
        if fs::metadata(&path).unwrap().is_dir() {
            continue;
        }
        match path.extension() {
            Some(name) if name == extension => {
                match path.file_name() {
                    Some(name) => {
                        try!(fs::copy(&path, destination.join(name)));
                    },
                    _ => {},
                }
            },
            _ => {},
        }
    }
    Ok(())
}
