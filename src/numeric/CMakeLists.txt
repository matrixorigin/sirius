target_sources(
  sirius_objects
  PRIVATE exact_decimal_gpu.cu
          decimal_types.cpp
          decimal_functions.cpp
          decimal_import.cpp
          decimal_aggregate_bind.cpp
          decimal_aggregate_gpu.cu
          decimal_aggregate_layout.cpp
          decimal_plan.cpp)
