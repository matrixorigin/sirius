add_library(sirius_embed STATIC src/embedding/c_api.cpp
                               src/embedding/runtime.cpp)
add_library(Sirius::embed ALIAS sirius_embed)
target_compile_features(sirius_embed PRIVATE cxx_std_20)
target_include_directories(sirius_embed PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(sirius_embed PUBLIC sirius_core duckdb_static
                                        dummy_static_extension_loader Threads::Threads)
install(FILES src/sirius_c.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

target_sources(sirius_objects PRIVATE src/embedding/control.cpp src/embedding/input.cpp
  src/embedding/native_gpu.cpp src/cuda/scan/native_mo_input.cu)

target_sources(sirius_objects PRIVATE src/embedding/plan_bindings.cpp)
target_sources(sirius_embed PRIVATE src/embedding/plan.cpp)
target_include_directories(sirius_embed PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tae-scanner/include
  ${SIRIUS_SUBSTRAIT_DIR}/third_party ${SIRIUS_SUBSTRAIT_DIR}/third_party/substrait)

target_sources(sirius_objects PRIVATE src/embedding/tae_demand.cpp)

target_sources(sirius_objects PRIVATE src/embedding/result.cpp src/embedding/result_codec.cpp src/embedding/result_gpu.cpp)

target_sources(sirius_objects PRIVATE src/embedding/execution_stats.cpp)
