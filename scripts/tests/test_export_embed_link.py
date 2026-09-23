# Copyright 2026 Sirius Contributors.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "export_embed_link", Path(__file__).resolve().parents[1] / "export_embed_link.py"
)
sdk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sdk)


class ExportTest(unittest.TestCase):
    def test_export_retires_stale_gpu_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "embedding-sdk"
            output.mkdir()
            stale = output / "toolchain.json"
            stale.write_text("obsolete GPU manifest")
            header = root / "sirius_c.h"
            header.write_text("#define SIRIUS_ABI_VERSION 1u\n")
            consumer = root / "consumer"
            consumer.write_bytes(b"consumer")
            argv = [
                "export_embed_link.py",
                "--ninja",
                "ninja",
                "--build",
                str(root),
                "--consumer",
                str(consumer),
                "--header",
                str(header),
                "--output",
                str(output),
            ]
            with patch.object(sys, "argv", argv), patch.object(
                sdk.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, "", ""),
            ), patch.object(
                sdk, "link_arguments", return_value=("/usr/bin/c++", [])
            ), patch.object(
                sdk, "c_compiler", return_value="/usr/bin/cc"
            ), patch.object(
                sdk, "source_provenance", return_value={"source_revision": "test"}
            ), patch.object(
                sdk, "artifact_hashes", return_value={}
            ):
                sdk.main()
            self.assertFalse(stale.exists())
            self.assertNotIn(
                "gpu_toolchain_manifest", json.loads((output / "link.json").read_text())
            )

    def test_c_compiler_is_verified_from_cache_and_actual_compile(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            compiler = build / "sdk-c-compiler"
            compiler.write_bytes(b"compiler fixture")
            compiler.chmod(0o755)
            (build / "CMakeCache.txt").write_text(
                "CMAKE_C_COMPILER:FILEPATH=" + str(compiler) + "\n"
            )
            for launcher in ("", "sccache "):
                command = (
                    launcher + str(compiler) + " -o obj/c_smoke.c.o -c /src/c_smoke.c"
                )
                self.assertEqual(sdk.c_compiler(command, build), str(compiler))
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                sdk.c_compiler(
                    "/usr/bin/cc -o obj/c_smoke.c.o -c /src/c_smoke.c", build
                )
            with self.assertRaisesRegex(RuntimeError, "exactly one verified"):
                sdk.c_compiler("/usr/bin/c++ -o consumer obj/c_smoke.c.o", build)

    def test_c_compiler_cache_must_name_an_absolute_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            for cache in (
                "CMAKE_C_COMPILER:STRING=/usr/bin/cc\n",
                "CMAKE_C_COMPILER:FILEPATH=cc\n",
                "CMAKE_C_COMPILER:FILEPATH=" + str(build / "missing") + "\n",
            ):
                with self.subTest(cache=cache):
                    (build / "CMakeCache.txt").write_text(cache)
                    with self.assertRaises(RuntimeError):
                        sdk.c_compiler("cc -o obj/c_smoke.c.o -c /src/c_smoke.c", build)

    def test_link_closure_preserves_archive_order_and_device_link(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            for name in ("embed.a", "duckdb.a", "device-link.o"):
                (build / name).write_bytes(name.encode())
            commands = ": && /usr/bin/c++ src/c_smoke.c.o device-link.o -o consumer -Wl,--start-group embed.a duckdb.a -Wl,--end-group -lcudart && :"
            compiler, flags = sdk.link_arguments(commands, build, build / "consumer")
            self.assertEqual(compiler, "/usr/bin/c++")
            self.assertEqual(
                flags,
                [
                    str(build / "device-link.o"),
                    "-Wl,--start-group",
                    str(build / "embed.a"),
                    str(build / "duckdb.a"),
                    "-Wl,--end-group",
                    "-lcudart",
                ],
            )

    def test_hashes_cover_header_consumer_static_device_and_shared(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                root / name
                for name in (
                    "sirius_c.h",
                    "consumer",
                    "embed.a",
                    "device-link.o",
                    "libgpu.so",
                )
            ]
            for path in paths:
                path.write_bytes(path.name.encode())
            loaded = f"libgpu.so => {paths[4]} (0x1)\nlibcuda.so.1 => /host/libcuda.so.1 (0x2)\nlibc.so.6 => /host/libc.so.6 (0x3)\nlibmvec.so.1 => /host/libmvec.so.1 (0x4)\n"
            with patch.object(
                sdk.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, loaded, ""),
            ):
                hashes = sdk.artifact_hashes(
                    [
                        "-isystem",
                        str(root),
                        str(paths[2]),
                        str(paths[3]),
                        "-L" + str(root),
                        "-lgpu",
                    ],
                    paths[1],
                    paths[0],
                    "c++",
                )
            self.assertEqual(set(hashes), {str(path) for path in paths})
            old = hashes[str(paths[3])]
            paths[3].write_bytes(b"changed device link")
            self.assertNotEqual(sdk.sha256(paths[3]), old)

    def test_source_provenance_uses_native_repository_and_marks_untracked(self):
        header = Path("/native/repo/src/include/sirius_c.h")
        results = [
            subprocess.CompletedProcess([], 0, "a" * 40 + "\n", ""),
            subprocess.CompletedProcess([], 0, "?? src/new.cc\n", ""),
        ]
        with patch.object(sdk.subprocess, "run", side_effect=results) as command:
            provenance = sdk.source_provenance(header)
        self.assertEqual(provenance["source_directory"], "/native/repo")
        self.assertEqual(provenance["source_revision"], "a" * 40)
        self.assertTrue(provenance["source_dirty"])
        self.assertIn("--untracked-files=normal", command.call_args.args[0])

    def test_opaque_response_and_unresolved_dependency_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "opaque"):
            sdk.link_arguments(
                ": && c++ src/c_smoke.c.o @opaque.rsp -o consumer && :",
                Path("/build"),
                Path("/build/consumer"),
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "consumer").write_bytes(b"c")
            (root / "header").write_bytes(b"h")
            with patch.object(
                sdk.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    [], 0, "libgpu.so => not found\n", ""
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "unresolved"):
                    sdk.artifact_hashes([], root / "consumer", root / "header", "c++")

    def test_static_l_inputs_are_in_the_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "libdevice.a").write_bytes(b"device library")
            self.assertEqual(
                sdk.link_files(["-L" + str(root), "-ldevice"], "c++"),
                {root / "libdevice.a"},
            )


if __name__ == "__main__":
    unittest.main()
