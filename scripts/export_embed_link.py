#!/usr/bin/env python3
# Copyright 2026 Sirius Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Export the verified C consumer's link closure, not a second hand-written list.

This is a build-tree SDK: absolute paths intentionally bind the metadata to the
native artifact generation. Relocatable MO runtime packaging is a later layer.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess


def sha256(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def source_provenance(header):
    source = header.resolve().parents[2]

    def git(*arguments):
        return subprocess.run(
            ["git", "-C", str(source), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "source_directory": str(source),
        "source_revision": git("rev-parse", "HEAD"),
        "source_dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
    }


def link_files(flags, compiler):
    paths = set()
    directories = []
    for index, flag in enumerate(flags):
        if flag == "-L":
            directories.append(Path(flags[index + 1]))
        elif flag.startswith("-L"):
            directories.append(Path(flag[2:]))
        elif not flag.startswith("-"):
            if index and flags[index - 1] in (
                "-isystem",
                "-I",
                "-L",
                "-iquote",
                "--sysroot",
                "-isysroot",
            ):
                continue
            path = Path(flag)
            if not path.is_file():
                raise RuntimeError("missing native link artifact: " + flag)
            paths.add(path.resolve())
    for flag in flags:
        if not flag.startswith("-l"):
            continue
        name = flag[2:]
        filenames = (
            [name[1:]]
            if name.startswith(":")
            else ["lib" + name + ".so", "lib" + name + ".a"]
        )
        selected = next(
            (
                directory / filename
                for directory in directories
                for filename in filenames
                if (directory / filename).is_file()
            ),
            None,
        )
        if selected is None:
            for filename in filenames:
                candidate = subprocess.run(
                    [compiler, "-print-file-name=" + filename],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                if candidate != filename and Path(candidate).is_file():
                    selected = Path(candidate)
                    break
        if selected is None:
            raise RuntimeError("cannot fingerprint native -l input: " + flag)
        paths.add(selected.resolve())
    return paths


def artifact_hashes(flags, consumer, header, compiler):
    """Hash actual ordered-link inputs, including archives and device-link objects.

    Shared dependencies selected through -l are resolved by the verified
    consumer's loader, including transitive libraries absent from link flags.
    Host glibc and the driver are deployment ABI inputs, not shipped SDK files.
    """
    paths = {consumer.resolve(), header.resolve()} | link_files(flags, compiler)
    dependencies = subprocess.run(
        ["ldd", str(consumer)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    host = re.compile(
        r"^(?:lib(?:cuda|nvidia-ml)\.so(?:\..*)?|lib(?:c|m|mvec|dl|rt|pthread|resolv|util)\.so(?:\..*)?|ld-linux.*)$"
    )
    for line in dependencies.splitlines():
        if "not found" in line:
            raise RuntimeError("unresolved native consumer dependency: " + line.strip())
        match = re.match(r"\s*(\S+)\s+=>\s+(/\S+)\s+\(", line)
        if match and not host.fullmatch(match[1]):
            paths.add(Path(match[2]).resolve())
    return {str(path): sha256(path) for path in sorted(paths)}


def link_arguments(commands, build, consumer):
    for line in reversed(commands.splitlines()):
        words = shlex.split(line)
        if "-o" not in words:
            continue
        output = words.index("-o")
        if (build / words[output + 1]).resolve() != consumer:
            continue
        # CMake's Ninja rule starts with ': && compiler' and ends with '&&'.
        start = words.index("&&") + 1 if words[:1] == [":"] else 0
        end = words.index("&&", output) if "&&" in words[output:] else len(words)
        compiler = words[start]
        arguments = []
        main_objects = 0
        for index in range(start + 1, end):
            word = words[index]
            if index in (output, output + 1):
                continue
            if word.endswith("/c_smoke.c.o"):
                main_objects += 1
                continue
            if word.startswith("-Wl,--dependency-file="):
                continue
            if word.startswith("@"):
                raise RuntimeError(
                    "opaque Ninja response file: cannot export a complete SDK"
                )
            if not word.startswith("-") and (build / word).exists():
                word = str((build / word).resolve())
            arguments.append(word)
        if main_objects != 1:
            raise RuntimeError("expected exactly one C smoke object in the native link")
        return compiler, arguments
    raise RuntimeError("no verified C consumer link command found")


def c_compiler(commands, build):
    matches = re.findall(
        r"^CMAKE_C_COMPILER:FILEPATH=(.+)$",
        (build / "CMakeCache.txt").read_text(),
        re.M,
    )
    if len(matches) != 1:
        raise RuntimeError("SDK requires exactly one CMAKE_C_COMPILER:FILEPATH")
    compiler = Path(matches[0])
    if (
        not compiler.is_absolute()
        or not compiler.is_file()
        or not os.access(compiler, os.X_OK)
    ):
        raise RuntimeError("SDK C compiler must be an existing absolute executable")
    verified = 0
    for line in commands.splitlines():
        words = shlex.split(line)
        if "-c" not in words or "-o" not in words:
            continue
        output = words.index("-o")
        if output + 1 >= len(words) or not words[output + 1].endswith("/c_smoke.c.o"):
            continue
        compile_at = words.index("-c")
        if (
            compile_at + 1 >= len(words)
            or Path(words[compile_at + 1]).name != "c_smoke.c"
        ):
            raise RuntimeError("C smoke object does not compile the expected C source")
        # A launcher such as sccache may precede the configured compiler. Use
        # the exact CMake cache path; do not derive gcc/cc from a C++ basename.
        if str(compiler) not in words[: min(compile_at, output)]:
            raise RuntimeError(
                "CMake C compiler does not match the C smoke compile command"
            )
        verified += 1
    if verified != 1:
        raise RuntimeError("SDK requires exactly one verified C smoke compile command")
    return str(compiler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ninja", required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--consumer", type=Path, required=True)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build = args.build.resolve()
    commands = subprocess.run(
        [args.ninja, "-C", str(build), "-t", "commands", "sirius_c_smoke"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    compiler, flags = link_arguments(commands, build, args.consumer.resolve())
    c_compiler_path = c_compiler(commands, build)
    match = re.search(
        r"^#define\s+SIRIUS_ABI_VERSION\s+(\d+)[uU]?\b", args.header.read_text(), re.M
    )
    if not match:
        raise RuntimeError("missing native ABI version")
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.header, args.output / "sirius_c.h")
    manifest = {
        "schema_version": 1,
        "abi_version": int(match[1]),
        "compiler": compiler,
        "c_compiler": c_compiler_path,
        "build_directory": str(build),
        "link_arguments": flags,
        "consumer": str(args.consumer.resolve()),
        **source_provenance(args.header),
        "artifact_sha256": artifact_hashes(
            flags, args.consumer.resolve(), args.header, compiler
        ),
    }
    manifest["artifact_sha256"][str((args.output / "sirius_c.h").resolve())] = sha256(
        args.output / "sirius_c.h"
    )
    (args.output / "link.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Reusing a build tree must not leave the retired GPU-provider manifest
    # beside a newly exported SDK that no longer records or validates it.
    (args.output / "toolchain.json").unlink(missing_ok=True)
    # GCC/Clang response files accept quoted arguments with backslash escapes.
    response = "\n".join(
        '"' + flag.replace("\\", "\\\\").replace('"', '\\"') + '"' for flag in flags
    )
    (args.output / "link.rsp").write_text(response + "\n")
    (args.output / "SiriusEmbedConfig.cmake").write_text(
        "if(NOT TARGET Sirius::embed)\n"
        "  add_library(Sirius::embed INTERFACE IMPORTED)\n"
        "  set_target_properties(Sirius::embed PROPERTIES\n"
        '    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}"\n'
        '    INTERFACE_LINK_OPTIONS "@${CMAKE_CURRENT_LIST_DIR}/link.rsp")\n'
        "endif()\n"
    )


if __name__ == "__main__":
    main()
