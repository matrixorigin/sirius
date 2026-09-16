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
