# Control/credit tests need neither a GPU nor DuckDB initialization. The C
# consumer below separately proves the real static dependency/link boundary.
add_executable(sirius_native_control_unittest
  test/cpp/exec/execution_evidence_unittest.cpp
  test/cpp/embedding/test_native_control.cpp src/embedding/control.cpp)
target_compile_features(sirius_native_control_unittest PRIVATE cxx_std_20)
target_include_directories(sirius_native_control_unittest PRIVATE
  "${SIRIUS_DUCKDB_SOURCE_DIR}/third_party/catch" "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(sirius_native_control_unittest PRIVATE Threads::Threads)
add_executable(sirius_c_smoke test/cpp/embedding/c_smoke.c)
set_target_properties(sirius_c_smoke PROPERTIES C_STANDARD 99 LINKER_LANGUAGE CXX)
target_link_libraries(sirius_c_smoke PRIVATE Sirius::embed)
set_property(TARGET sirius_c_smoke APPEND PROPERTY LINK_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/export_embed_link.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/sirius_c.h")
add_custom_command(TARGET sirius_c_smoke POST_BUILD
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/export_embed_link.py"
    --ninja "${CMAKE_MAKE_PROGRAM}" --build "${CMAKE_BINARY_DIR}"
    --consumer "$<TARGET_FILE:sirius_c_smoke>"
    --header "${CMAKE_CURRENT_SOURCE_DIR}/src/sirius_c.h"
    --output "${CMAKE_CURRENT_BINARY_DIR}/embedding-sdk"
  VERBATIM)

