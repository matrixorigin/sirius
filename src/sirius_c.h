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

/* create copies the bounded plan; prepare queues it on the native coordinator.
 * A timeout from prepare/wait does not cancel the query. An unstarted prepared
 * query is still subject to its end-to-end deadline. start is single-use.
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
sirius_status sirius_query_start(sirius_query_handle* query, sirius_error* error);
sirius_status sirius_query_cancel(sirius_query_handle* query, sirius_error* error);
sirius_status sirius_query_wait(sirius_query_handle* query, uint32_t wait_ms, sirius_error* error);
sirius_status sirius_query_close(sirius_query_handle** query,
                                 uint32_t wait_ms,
                                 sirius_error* error);

#ifdef __cplusplus
}
#endif
#endif
