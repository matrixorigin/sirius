target_sources(sirius_objects PRIVATE
  src/op/scan/tae_scan_plan.cpp
  src/op/scan/tae_gpu_ingestible.cpp
  src/data/host_tae_representation.cpp
  src/data/host_tae_representation_converters.cpp
  src/tae/tae_metadata_parser.cpp
  src/cuda/tae/tae_decode_fixed.cu
  src/cuda/tae/tae_decode_varchar.cu
  src/cuda/tae/tae_null_mask.cu)
target_include_directories(sirius_objects PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tae-scanner/include)
foreach(target sirius_objects sirius_core sirius_shared)
  target_link_libraries(${target} PRIVATE nvcomp tae_scanner_extension)
endforeach()
