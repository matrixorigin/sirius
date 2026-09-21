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

typedef struct pb_buffer {
  unsigned char data[256];
  size_t size;
} pb_buffer;

static int pb_varint(pb_buffer* out, uint64_t value)
{
  do {
    if (out->size == sizeof(out->data)) return 0;
    unsigned char byte = (unsigned char)(value & 0x7fu);
    value >>= 7;
    out->data[out->size++] = (unsigned char)(byte | (value ? 0x80u : 0u));
  } while (value);
  return 1;
}

static int pb_uint(pb_buffer* out, uint32_t field, uint64_t value)
{
  return pb_varint(out, ((uint64_t)field << 3) | 0u) && pb_varint(out, value);
}

static int pb_bytes(pb_buffer* out, uint32_t field, const void* bytes, size_t count)
{
  if (!pb_varint(out, ((uint64_t)field << 3) | 2u) || !pb_varint(out, count) ||
      count > sizeof(out->data) - out->size)
    return 0;
  memcpy(out->data + out->size, bytes, count);
  out->size += count;
  return 1;
}

static int pb_text(pb_buffer* out, uint32_t field, const char* text)
{
  return pb_bytes(out, field, text, strlen(text));
}

/* One Substrait 0.78 root reading registered binding 1 as one required BIGINT column. */
static int make_bound_plan(pb_buffer* plan)
{
  pb_buffer i64 = {{0}, 0}, type = {{0}, 0}, types = {{0}, 0};
  pb_buffer schema = {{0}, 0}, table = {{0}, 0}, read = {{0}, 0};
  pb_buffer rel = {{0}, 0}, root = {{0}, 0}, plan_rel = {{0}, 0}, version = {{0}, 0};
  memset(plan, 0, sizeof(*plan));
  return pb_uint(&i64, 2, 2) && pb_bytes(&type, 7, &i64, i64.size) &&
         pb_bytes(&types, 1, &type, type.size) && pb_text(&schema, 1, "c") &&
         pb_bytes(&schema, 2, &types, types.size) && pb_text(&table, 1, "__sirius_embedded_v1") &&
         pb_text(&table, 1, "1") && pb_bytes(&read, 2, &schema, schema.size) &&
         pb_bytes(&read, 7, &table, table.size) && pb_bytes(&rel, 1, &read, read.size) &&
         pb_bytes(&root, 1, &rel, rel.size) && pb_text(&root, 2, "c") &&
         pb_bytes(&plan_rel, 2, &root, root.size) && pb_uint(&version, 2, 78) &&
         pb_bytes(plan, 6, &version, version.size) && pb_bytes(plan, 3, &plan_rel, plan_rel.size);
}

int main(int argc, char** argv)
{
  int result                   = 0;
  sirius_engine_handle* engine = NULL;
  sirius_engine_handle* second = NULL;
  sirius_query_handle* query   = NULL;
  sirius_input_handle* input   = NULL;
  sirius_batch_handle* batch   = NULL;
  sirius_error error;
  pb_buffer plan;
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
    CHECK(make_bound_plan(&plan));
    CHECK(sirius_query_create(engine, &qopts, plan.data, plan.size, &query, &error) == SIRIUS_OK);
    {
      sirius_input_column column     = {23, 0, 0, 0};
      sirius_column logical          = {23, 0, 0, 0, "c", 1, 0};
      sirius_read_column read_column = {logical, 7, 3, 0};
      sirius_query_contract contract = {
        sizeof(contract), SIRIUS_ABI_VERSION, 0, 0, "q", 1, {0}, &logical, 1};
      sirius_read_binding binding = {sizeof(binding),
                                     SIRIUS_ABI_VERSION,
                                     1,
                                     SIRIUS_READ_MO,
                                     0,
                                     "db",
                                     2,
                                     "t",
                                     1,
                                     "s",
                                     1,
                                     &read_column,
                                     1,
                                     NULL,
                                     0,
                                     NULL,
                                     0};
      CHECK(sirius_query_bind(query, &contract, &error) == SIRIUS_OK);
      CHECK(sirius_read_register(query, &binding, &error) == SIRIUS_OK);
      CHECK(sirius_input_register(query, 1, &column, 1, &input, &error) == SIRIUS_OK);
      CHECK(sirius_input_acquire(input, 8, 0, &batch, &error) == SIRIUS_INVALID_STATE);
      CHECK(batch == NULL);
    }
    CHECK(sirius_query_prepare(query, 10000, &error) == SIRIUS_OK);
    CHECK(sirius_input_acquire(input, 8, 0, &batch, &error) == SIRIUS_INVALID_STATE);
    CHECK(batch == NULL);
    CHECK(sirius_query_start(query, &error) == SIRIUS_UNSUPPORTED);
    CHECK(sirius_query_cancel(query, &error) == SIRIUS_OK);
    CHECK(sirius_query_wait(query, 10000, &error) == SIRIUS_CANCELLED);
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
