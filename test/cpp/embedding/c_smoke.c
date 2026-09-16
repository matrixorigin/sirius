/* Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "sirius_c.h"

#include <stdio.h>
#include <string.h>

#define CHECK(x)                                          \
  do {                                                    \
    if (!(x)) {                                           \
      fprintf(stderr, "failed: %s (%d)\n", #x, __LINE__); \
      result = 1;                                         \
      goto cleanup;                                       \
    }                                                     \
  } while (0)

int main(int argc, char** argv)
{
  int result                   = 0;
  sirius_engine_handle* engine = NULL;
  sirius_engine_handle* second = NULL;
  sirius_query_handle* query   = NULL;
  sirius_input_handle* input   = NULL;
  sirius_batch_handle* batch   = NULL;
  sirius_error error;
  sirius_engine_options options = {sizeof(options), SIRIUS_ABI_VERSION, NULL, 0, 0};
  CHECK(sirius_abi_version() == SIRIUS_ABI_VERSION);
  CHECK((sirius_capabilities() & SIRIUS_CAP_ENGINE_CONTROL) != 0);
  CHECK(sirius_engine_create(NULL, &engine, &error) == SIRIUS_INVALID_ARGUMENT);
  CHECK(engine == NULL && error.code == SIRIUS_INVALID_ARGUMENT);
  CHECK(sirius_engine_close(&engine, 0, &error) == SIRIUS_OK);
  CHECK(sirius_query_close(&query, 0, &error) == SIRIUS_OK);
  if (argc == 2) {
    sirius_query_options qopts = {sizeof(qopts), SIRIUS_ABI_VERSION, 10000, 0};
    sirius_engine_stats stats  = {sizeof(stats), SIRIUS_ABI_VERSION, 0, 0, 0, 0};
    options.config_path        = argv[1];
    options.config_path_bytes  = (uint32_t)strlen(argv[1]);
    if (sirius_engine_create(&options, &engine, &error) != SIRIUS_OK) {
      fprintf(stderr, "engine create: %s\n", error.message);
      result = 1;
      goto cleanup;
    }
    CHECK(sirius_engine_get_stats(engine, &stats, &error) == SIRIUS_OK);
    CHECK(stats.accepting_queries && stats.live_queries == 0);
    CHECK(sirius_engine_create(&options, &engine, &error) == SIRIUS_INVALID_ARGUMENT);
    CHECK(engine != NULL);
    CHECK(sirius_engine_create(&options, &second, &error) == SIRIUS_BUSY);
    CHECK(second == NULL);
    CHECK(sirius_query_create(engine, &qopts, "x", 1, &query, &error) == SIRIUS_OK);
    {
      sirius_input_column column = {23, 0, 0, 0};
      CHECK(sirius_input_register(query, 1, &column, 1, &input, &error) == SIRIUS_OK);
      CHECK(sirius_input_acquire(input, 8, 0, &batch, &error) == SIRIUS_INVALID_STATE);
      CHECK(batch == NULL);
    }
    CHECK(sirius_query_prepare(query, 10000, &error) == SIRIUS_UNSUPPORTED);
    CHECK(sirius_query_wait(query, 10000, &error) == SIRIUS_UNSUPPORTED);
    CHECK(sirius_query_start(query, &error) == SIRIUS_INVALID_STATE);
    CHECK(sirius_query_close(&query, 10000, &error) == SIRIUS_BUSY && query != NULL);
    CHECK(sirius_input_close(&input, &error) == SIRIUS_OK && input == NULL);
    CHECK(sirius_query_close(&query, 10000, &error) == SIRIUS_OK && query == NULL);
    CHECK(sirius_query_create(engine, &qopts, "x", 1, &query, &error) == SIRIUS_OK);
    CHECK(sirius_query_create(engine, &qopts, "x", 1, &query, &error) == SIRIUS_INVALID_ARGUMENT);
    CHECK(query != NULL);
    CHECK(sirius_query_cancel(query, &error) == SIRIUS_OK);
    CHECK(sirius_query_wait(query, 10000, &error) == SIRIUS_CANCELLED);
    CHECK(sirius_engine_close(&engine, 10000, &error) == SIRIUS_BUSY);
    CHECK(engine != NULL);
    CHECK(sirius_query_close(&query, 10000, &error) == SIRIUS_OK && query == NULL);
    CHECK(sirius_engine_close(&engine, 10000, &error) == SIRIUS_OK && engine == NULL);
    /* A completed shutdown releases the process-wide runtime guard. */
    options.gpu_streams = 1;
    CHECK(sirius_engine_create(&options, &engine, &error) == SIRIUS_OK);
    CHECK(sirius_engine_close(&engine, 10000, &error) == SIRIUS_OK && engine == NULL);
  }
cleanup:
  if (batch && sirius_batch_release(&batch, &error) != SIRIUS_OK) result = 1;
  if (input && sirius_input_close(&input, &error) != SIRIUS_OK) result = 1;
  if (second && sirius_engine_close(&second, 10000, &error) != SIRIUS_OK) result = 1;
  if (query && sirius_query_close(&query, 10000, &error) != SIRIUS_OK) result = 1;
  if (engine && sirius_engine_close(&engine, 10000, &error) != SIRIUS_OK) result = 1;
  return result;
}
