/* Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef SIRIUS_C_H
#define SIRIUS_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SIRIUS_ABI_VERSION         1u
#define SIRIUS_ERROR_MESSAGE_BYTES 512u

typedef struct sirius_engine_handle sirius_engine_handle;
typedef struct sirius_query_handle sirius_query_handle;
typedef struct sirius_batch_handle sirius_batch_handle;
typedef struct sirius_input_handle sirius_input_handle;

typedef struct sirius_column {
  uint32_t oid;
  int32_t width;
  int32_t scale;
  uint32_t nullable;
  const char* name;
  uint32_t name_bytes;
  uint32_t reserved;
} sirius_column;

typedef struct sirius_read_column {
  sirius_column logical;
  uint64_t physical_column_id;
  uint32_t sequence_number;
  uint32_t reserved;
} sirius_read_column;

enum { SIRIUS_READ_MO = 1, SIRIUS_READ_TAE = 2 };

typedef struct sirius_query_contract {
  uint32_t struct_size, abi_version;
  uint32_t account_id;
  uint32_t reserved;
  const char* query_id;
  uint32_t query_id_bytes;
  uint8_t snapshot_ts[12];
  const sirius_column* output_columns;
  uint32_t output_column_count;
} sirius_query_contract;

typedef struct sirius_read_binding {
  uint32_t struct_size, abi_version;
  uint64_t binding_id;
  uint32_t source_kind;
  uint32_t reserved;
  const char* database_name;
  uint32_t database_name_bytes;
  const char* table_name;
  uint32_t table_name_bytes;
  const char* schema_name;
  uint32_t schema_name_bytes;
  const sirius_read_column* columns;
  uint32_t column_count;
  const void* tae_manifest;
  uint64_t tae_manifest_bytes;
  const char* data_root;
  uint32_t data_root_bytes;
} sirius_read_binding;

/* MO native scalar OIDs; layout is little-endian, with 24-byte MO varlena.
 * Only the scalar types documented in embedding-c-api.md are accepted. */
typedef struct sirius_input_column {
  uint32_t oid;
  int32_t width;
  int32_t scale;
  uint32_t nullable;
} sirius_input_column;

enum { SIRIUS_VECTOR_FLAT = 0, SIRIUS_VECTOR_CONSTANT = 1, SIRIUS_VECTOR_NULL = 2 };
typedef struct sirius_input_vector {
  uint32_t vector_class;
  uint32_t reserved;
  uint64_t data_offset, data_bytes;
  uint64_t area_offset, area_bytes;
  /* Raw little-endian null words: 1 means NULL; omitted trailing words are zero. */
  uint64_t null_offset, null_bytes;
} sirius_input_vector;

typedef struct sirius_input_stats {
  uint32_t struct_size, abi_version;
  uint64_t retained_bytes, peak_bytes, leases, queued_batches, filling_batches;
  uint64_t blocked_acquires, source_units;
} sirius_input_stats;

/* Borrowed schema metadata remains valid until the query is successfully closed. */
typedef struct sirius_result_schema {
  uint32_t struct_size, abi_version, column_count, reserved;
  const sirius_column* columns;
} sirius_result_schema;

/* Column descriptors remain valid until batch_release. Payload is native-owned
 * and may be segmented; use result_read for synchronous bulk copies. */
typedef struct sirius_result_batch_info {
  uint32_t struct_size, abi_version, rows, column_count;
  uint64_t payload_bytes;
  const sirius_input_vector* columns;
} sirius_result_batch_info;

typedef struct sirius_result_stats {
  uint32_t struct_size, abi_version;
  uint64_t retained_bytes, peak_bytes, leases, queued_batches;
  uint64_t borrowed_batches, filling_batches, parked_publications, blocked_publications;
} sirius_result_stats;

typedef uint32_t sirius_status;
enum {
  SIRIUS_OK                 = 0,
  SIRIUS_UNSUPPORTED        = 1,
  SIRIUS_INVALID_ARGUMENT   = 2,
  SIRIUS_INVALID_STATE      = 3,
  SIRIUS_RESOURCE_EXHAUSTED = 4,
  SIRIUS_CANCELLED          = 5,
  SIRIUS_TIMEOUT            = 6,
  SIRIUS_GPU_UNAVAILABLE    = 7,
  SIRIUS_EOF                = 8,
  SIRIUS_NOT_NEEDED         = 9,
  SIRIUS_EXECUTION_FAILED   = 10,
  SIRIUS_BUSY               = 11
};

enum {
  SIRIUS_CAP_ENGINE_CONTROL = 1u,
  SIRIUS_CAP_MO_INPUT       = 2u,
  SIRIUS_CAP_TAE_INPUT      = 4u,
  SIRIUS_CAP_NATIVE_RESULTS = 8u
};

typedef struct sirius_error {
  sirius_status code;
  char message[SIRIUS_ERROR_MESSAGE_BYTES];
} sirius_error;

typedef struct sirius_engine_options {
  uint32_t struct_size;
  uint32_t abi_version;
  /* Explicit config; copied by create. No process-environment mutation. */
  const char* config_path;
  uint32_t config_path_bytes;
  /* Zero selects 16; values above 16 are rejected. */
  uint32_t max_waiting_queries;
  /* Zero selects two GPU pipeline workers; one is supported, not imposed. */
  uint32_t gpu_streams;
  uint32_t reserved;
} sirius_engine_options;

typedef struct sirius_query_options {
  uint32_t struct_size;
  uint32_t abi_version;
  /* End-to-end preparation/start/execution deadline; zero selects 15 minutes. */
  uint32_t timeout_ms;
  uint32_t reserved;
} sirius_query_options;

typedef struct sirius_engine_stats {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t accepting_queries;
  uint32_t unavailable;
  uint64_t live_queries;
  uint64_t queued_queries;
} sirius_engine_stats;

/* All entry points catch C++ exceptions. Error storage is caller-owned, may be
 * NULL, and is overwritten on each call. Create outputs must initially be NULL;
 * a rejected create never overwrites an existing live handle. Handles must not
 * be used after close.
 * Close requires exclusive ownership of the handle itself: cancel/wait calls
 * made by other threads must return before the caller destroys their handle.
 * Nonzero close results leave the handle valid so cleanup can be retried.
 */
uint32_t sirius_abi_version(void);
uint64_t sirius_capabilities(void);
sirius_status sirius_engine_create(const sirius_engine_options* options,
                                   sirius_engine_handle** out,
                                   sirius_error* error);
sirius_status sirius_engine_get_stats(sirius_engine_handle* engine,
                                      sirius_engine_stats* out,
                                      sirius_error* error);
sirius_status sirius_engine_stop(sirius_engine_handle* engine, sirius_error* error);
sirius_status sirius_engine_close(sirius_engine_handle** engine,
                                  uint32_t wait_ms,
                                  sirius_error* error);

/* create copies the bounded plan; bind/read_register copy their descriptors while
 * CREATED. prepare queues the plan on the native coordinator. A timeout from
 * prepare/wait does not cancel the query. An unstarted prepared query is still
 * subject to its end-to-end deadline. start is single-use after it succeeds.
 */
sirius_status sirius_query_create(sirius_engine_handle* engine,
                                  const sirius_query_options* options,
                                  const void* plan,
                                  uint64_t plan_bytes,
                                  sirius_query_handle** out,
                                  sirius_error* error);
sirius_status sirius_query_prepare(sirius_query_handle* query,
                                   uint32_t wait_ms,
                                   sirius_error* error);
sirius_status sirius_query_bind(sirius_query_handle* query,
                                const sirius_query_contract* contract,
                                sirius_error* error);
sirius_status sirius_read_register(sirius_query_handle* query,
                                   const sirius_read_binding* binding,
                                   sirius_error* error);
sirius_status sirius_query_start(sirius_query_handle* query, sirius_error* error);
sirius_status sirius_query_cancel(sirius_query_handle* query, sirius_error* error);
sirius_status sirius_query_wait(sirius_query_handle* query, uint32_t wait_ms, sirius_error* error);
sirius_status sirius_query_close(sirius_query_handle** query,
                                 uint32_t wait_ms,
                                 sirius_error* error);

/* Prepare exposes the schema; start enables result retrieval. There is one
 * pull consumer per query, independent of concurrent cancellation. A call
 * timeout does not cancel execution. EOF follows successful quiescence and is
 * distinct from an empty batch. Returned leases survive cancellation; close
 * remains BUSY until they are released. */
sirius_status sirius_query_get_schema(sirius_query_handle* query,
                                      sirius_result_schema* out,
                                      sirius_error* error);
sirius_status sirius_query_next_result(sirius_query_handle* query,
                                       uint32_t wait_ms,
                                       sirius_batch_handle** out,
                                       sirius_error* error);
sirius_status sirius_result_describe(sirius_batch_handle* batch,
                                     sirius_result_batch_info* out,
                                     sirius_error* error);
sirius_status sirius_result_read(sirius_batch_handle* batch,
                                 uint64_t offset,
                                 void* destination,
                                 uint64_t bytes,
                                 sirius_error* error);
sirius_status sirius_query_get_result_stats(sirius_query_handle* query,
                                            sirius_result_stats* out,
                                            sirius_error* error);

/* Register while CREATED; acquire only after successful startable preparation. Each read
 * has a strict 64 MiB window, including filling, queued and GPU/retry owners.
 * Close/release require exclusive ownership of the affected handle. */
sirius_status sirius_input_register(sirius_query_handle* query,
                                    uint64_t binding_id,
                                    const sirius_input_column* columns,
                                    uint32_t column_count,
                                    sirius_input_handle** out,
                                    sirius_error* error);
sirius_status sirius_input_acquire(sirius_input_handle* input,
                                   uint64_t bytes,
                                   uint32_t wait_ms,
                                   sirius_batch_handle** out,
                                   sirius_error* error);
sirius_status sirius_input_write(sirius_batch_handle* batch,
                                 uint64_t offset,
                                 const void* data,
                                 uint64_t bytes,
                                 sirius_error* error);
/* Publish copies descriptors. Success consumes *batch; failure preserves it.
 * A zero-row batch is not EOF. Caller pointers are never retained. */
sirius_status sirius_input_publish(sirius_input_handle* input,
                                   sirius_batch_handle** batch,
                                   uint32_t rows,
                                   const sirius_input_vector* columns,
                                   uint32_t column_count,
                                   sirius_error* error);
sirius_status sirius_input_finish(sirius_input_handle* input, sirius_error* error);
sirius_status sirius_input_fail(sirius_input_handle* input,
                                const char* message,
                                uint32_t message_bytes,
                                sirius_error* error);
sirius_status sirius_input_get_stats(sirius_input_handle* input,
                                     sirius_input_stats* out,
                                     sirius_error* error);
sirius_status sirius_input_close(sirius_input_handle** input, sirius_error* error);
sirius_status sirius_batch_release(sirius_batch_handle** batch, sirius_error* error);

#ifdef __cplusplus
}
#endif
#endif
