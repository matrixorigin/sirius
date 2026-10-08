# Embedding control tests need neither a GPU nor a DuckDB runtime. Keep them
# independently runnable from the linked ABI and GPU integration tests.
add_executable(
  sirius_exact_decimal_unittest test/cpp/exec/execution_evidence_unittest.cpp
                                test/cpp/embedding/test_exact_decimal.cpp)
target_compile_features(sirius_exact_decimal_unittest PRIVATE cxx_std_20)
target_include_directories(
  sirius_exact_decimal_unittest PRIVATE "${CMAKE_SOURCE_DIR}/third_party/catch"
                                        "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_executable(sirius_exact_decimal_benchmark EXCLUDE_FROM_ALL
               bench/mo_exact_decimal.cpp)
target_compile_features(sirius_exact_decimal_benchmark PRIVATE cxx_std_20)
target_include_directories(sirius_exact_decimal_benchmark
                           PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(sirius_exact_decimal_benchmark PRIVATE sirius_extension)

add_executable(
  sirius_native_control_unittest
  test/cpp/exec/execution_evidence_unittest.cpp
  test/cpp/embedding/test_native_control.cpp
  test/cpp/embedding/test_native_execution_stats.cpp
  test/cpp/exec/test_multi_index_priority_queue.cpp
  test/cpp/embedding/test_native_input.cpp
  test/cpp/embedding/test_tae_demand.cpp
  test/cpp/embedding/test_native_result.cpp
  src/embedding/control.cpp
  src/embedding/execution_stats.cpp
  src/embedding/input.cpp
  src/embedding/tae_demand.cpp
  src/embedding/result.cpp)
target_compile_features(sirius_native_control_unittest PRIVATE cxx_std_20)
target_include_directories(
  sirius_native_control_unittest PRIVATE "${CMAKE_SOURCE_DIR}/third_party/catch"
                                         "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(sirius_native_control_unittest PRIVATE Threads::Threads)

add_executable(sirius_native_result_integration
               test/cpp/embedding/native_result_integration.cpp)
target_compile_features(sirius_native_result_integration PRIVATE cxx_std_20)
target_include_directories(
  sirius_native_result_integration
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
          "${SIRIUS_SUBSTRAIT_DIR}/third_party"
          "${SIRIUS_SUBSTRAIT_DIR}/third_party/substrait")
target_link_libraries(sirius_native_result_integration PRIVATE Sirius::embed
                                                               Threads::Threads)

add_executable(
  sirius_native_binding_unittest
  test/cpp/exec/execution_evidence_unittest.cpp
  test/cpp/embedding/test_native_binding.cpp src/embedding/plan.cpp)
target_compile_features(sirius_native_binding_unittest PRIVATE cxx_std_20)
target_include_directories(
  sirius_native_binding_unittest
  PRIVATE "${CMAKE_SOURCE_DIR}/third_party/catch"
          "${CMAKE_CURRENT_SOURCE_DIR}/src"
          "${CMAKE_CURRENT_SOURCE_DIR}/tae-scanner/include"
          "${SIRIUS_SUBSTRAIT_DIR}/third_party"
          "${SIRIUS_SUBSTRAIT_DIR}/third_party/substrait")
target_link_libraries(sirius_native_binding_unittest PRIVATE sirius_extension
                                                             Threads::Threads)

add_executable(
  sirius_native_gpu_unittest
  test/cpp/exec/execution_evidence_unittest.cpp
  test/cpp/embedding/test_native_gpu.cpp
  test/cpp/embedding/test_native_admission.cpp
  test/cpp/embedding/test_native_result_codec.cpp
  test/cpp/embedding/test_exact_decimal_gpu.cpp
  test/cpp/embedding/test_tae_gpu.cpp
  src/embedding/c_api.cpp)
target_compile_features(sirius_native_gpu_unittest PRIVATE cxx_std_20)
target_include_directories(
  sirius_native_gpu_unittest
  PRIVATE "${CMAKE_SOURCE_DIR}/third_party/catch"
          "${CMAKE_CURRENT_SOURCE_DIR}/test/cpp"
          "${CMAKE_CURRENT_SOURCE_DIR}/tae-scanner/include"
          "${CMAKE_CURRENT_SOURCE_DIR}/src"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/compression/simpatico_codegen/src")
target_compile_definitions(
  sirius_native_gpu_unittest
  PRIVATE SIRIUS_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(
  sirius_native_gpu_unittest
  PRIVATE sirius_extension duckdb_static dummy_static_extension_loader
          Threads::Threads)

# Keep the scanner bind-data test's private include edge explicit.
set_source_files_properties(
  test/cpp/integration/test_gpu_execution_tae_scan.cpp
  PROPERTIES INCLUDE_DIRECTORIES
             "${CMAKE_CURRENT_SOURCE_DIR}/tae-scanner/include")
