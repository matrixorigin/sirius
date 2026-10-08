# The C ABI owns only its opaque handles and links the upstream Sirius object
# closure through the static extension target. The verified C consumer below
# exports exact C/C++ compiler and link inputs for MatrixOne.
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_library(
  sirius_embed STATIC src/embedding/c_api.cpp src/embedding/runtime.cpp
                      src/embedding/plan.cpp)
add_library(Sirius::embed ALIAS sirius_embed)
target_compile_features(sirius_embed PRIVATE cxx_std_20)
target_include_directories(
  sirius_embed
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tae-scanner/include
          ${SIRIUS_SUBSTRAIT_DIR}/third_party
          ${SIRIUS_SUBSTRAIT_DIR}/third_party/substrait)
target_link_libraries(
  sirius_embed PUBLIC sirius_extension duckdb_static
                      dummy_static_extension_loader Threads::Threads)

add_executable(sirius_c_smoke test/cpp/embedding/c_smoke.c)
set_target_properties(sirius_c_smoke PROPERTIES C_STANDARD 99 LINKER_LANGUAGE
                                                              CXX)
target_link_libraries(sirius_c_smoke PRIVATE Sirius::embed)
set_property(
  TARGET sirius_c_smoke
  APPEND
  PROPERTY
    LINK_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/export_embed_link.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/proto/matrixone/sirius/numeric/v1/exact_decimal.proto"
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
