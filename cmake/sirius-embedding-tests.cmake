# Reuse the existing end-to-end TAE cases without rebuilding unrelated test
# operators.
add_executable(
  sirius_native_tae_unittest
  test/cpp/exec/execution_evidence_unittest.cpp
  test/cpp/integration/test_gpu_execution_tae_scan.cpp
  test/cpp/utils/sirius_test_env.cpp)
target_compile_features(sirius_native_tae_unittest PRIVATE cxx_std_20)
target_include_directories(
  sirius_native_tae_unittest
  PRIVATE
    "${SIRIUS_DUCKDB_SOURCE_DIR}/third_party/catch"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/compression/simpatico_codegen/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/tae-scanner/include")
target_compile_definitions(
  sirius_native_tae_unittest
  PRIVATE SIRIUS_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(sirius_native_tae_unittest sirius_core duckdb_static
                      Threads::Threads)
sirius_link_duckdb_extensions(sirius_native_tae_unittest)

# Control/credit tests need neither a GPU nor DuckDB initialization. The C
# consumer below separately proves the real static dependency/link boundary.
add_executable(
  sirius_native_control_unittest
  test/cpp/exec/execution_evidence_unittest.cpp
  test/cpp/embedding/test_native_control.cpp
  test/cpp/exec/test_multi_index_priority_queue.cpp
  test/cpp/embedding/test_native_input.cpp
  test/cpp/embedding/test_tae_demand.cpp
  src/embedding/control.cpp
  src/embedding/input.cpp
  src/embedding/tae_demand.cpp)
target_compile_features(sirius_native_control_unittest PRIVATE cxx_std_20)
target_include_directories(
  sirius_native_control_unittest
  PRIVATE "${SIRIUS_DUCKDB_SOURCE_DIR}/third_party/catch"
          "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(sirius_native_control_unittest PRIVATE Threads::Threads)
add_executable(sirius_c_smoke test/cpp/embedding/c_smoke.c)
add_executable(
  sirius_native_binding_unittest
  test/cpp/exec/execution_evidence_unittest.cpp
  test/cpp/embedding/test_native_binding.cpp src/embedding/plan.cpp)
target_compile_features(sirius_native_binding_unittest PRIVATE cxx_std_20)
target_include_directories(
  sirius_native_binding_unittest
  PRIVATE "${SIRIUS_DUCKDB_SOURCE_DIR}/third_party/catch"
          "${CMAKE_CURRENT_SOURCE_DIR}/src"
          "${CMAKE_CURRENT_SOURCE_DIR}/tae-scanner/include"
          "${SIRIUS_SUBSTRAIT_DIR}/third_party"
          "${SIRIUS_SUBSTRAIT_DIR}/third_party/substrait")
target_link_libraries(sirius_native_binding_unittest PRIVATE sirius_core
                                                             Threads::Threads)
add_executable(
  sirius_native_gpu_unittest
  test/cpp/exec/execution_evidence_unittest.cpp
  test/cpp/embedding/test_native_gpu.cpp
  test/cpp/embedding/test_native_admission.cpp
  test/cpp/embedding/test_tae_gpu.cpp
  src/embedding/c_api.cpp)
target_compile_features(sirius_native_gpu_unittest PRIVATE cxx_std_20)
target_include_directories(
  sirius_native_gpu_unittest
  PRIVATE
    "${SIRIUS_DUCKDB_SOURCE_DIR}/third_party/catch"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tae-scanner/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/compression/simpatico_codegen/include")
target_compile_definitions(
  sirius_native_gpu_unittest
  PRIVATE SIRIUS_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(
  sirius_native_gpu_unittest
  PRIVATE sirius_core duckdb_static dummy_static_extension_loader
          Threads::Threads)
set_target_properties(sirius_c_smoke PROPERTIES C_STANDARD 99 LINKER_LANGUAGE
                                                              CXX)
target_link_libraries(sirius_c_smoke PRIVATE Sirius::embed)
set_property(
  TARGET sirius_c_smoke
  APPEND
  PROPERTY LINK_DEPENDS
           "${CMAKE_CURRENT_SOURCE_DIR}/scripts/export_embed_link.py"
           "${CMAKE_CURRENT_SOURCE_DIR}/src/sirius_c.h")
add_custom_command(
  TARGET sirius_c_smoke
  POST_BUILD
  COMMAND
    "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/export_embed_link.py" --ninja
    "${CMAKE_MAKE_PROGRAM}" --build "${CMAKE_BINARY_DIR}" --consumer
    "$<TARGET_FILE:sirius_c_smoke>" --header
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sirius_c.h" --output
    "${CMAKE_CURRENT_BINARY_DIR}/embedding-sdk"
  VERBATIM)

