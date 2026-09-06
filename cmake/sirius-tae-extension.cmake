# Register the pinned TAE scanner alongside DuckDB's built-in extensions.
duckdb_extension_load(tae_scanner
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../tae-scanner
  INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../tae-scanner/include)
