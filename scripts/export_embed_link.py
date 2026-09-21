#!/usr/bin/env python3
# Copyright 2026 Sirius Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Export the verified C consumer's link closure, not a second hand-written list.

This is a build-tree SDK: absolute paths intentionally bind the metadata to the
native artifact generation. Relocatable MO runtime packaging is a later layer.
"""

import argparse
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess


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
        "build_directory": str(build),
        "link_arguments": flags,
    }
    (args.output / "link.json").write_text(json.dumps(manifest, indent=2) + "\n")
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
