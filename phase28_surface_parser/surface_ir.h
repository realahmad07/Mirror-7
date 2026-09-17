#ifndef MIRROR7_SURFACE_IR_H
#define MIRROR7_SURFACE_IR_H

#include <stddef.h>
#include <stdint.h>

#define MIRROR7_SURFACE_IR_VERSION 1u
#define MIRROR7_SURFACE_IR_MAX_TOKENS 4096u
#define MIRROR7_SURFACE_IR_MAX_NAME 63u

typedef enum {
    MIRROR7_IR_COLON = 1,
    MIRROR7_IR_NAME = 2,
    MIRROR7_IR_NUMBER = 3,
    MIRROR7_IR_SEMI = 4,
    MIRROR7_IR_IF = 5,
    MIRROR7_IR_ELSE = 6,
    MIRROR7_IR_THEN = 7
} mirror7_ir_kind_t;

typedef struct {
    uint8_t kind;
    uint16_t number;
    uint16_t length;
    uint32_t source_pos;
    char text[MIRROR7_SURFACE_IR_MAX_NAME + 1u];
} mirror7_ir_token_t;

typedef struct {
    uint16_t version;
    uint16_t token_count;
    mirror7_ir_token_t tokens[MIRROR7_SURFACE_IR_MAX_TOKENS];
} mirror7_surface_ir_t;

/*
 * ABI contract:
 *   parser owns construction of this value; compiler only consumes it.
 *   token order is source order; source_pos is the byte offset of the token.
 *   NUMBER uses the uint16_t number field; all other kinds use text.
 *   A successful parse always has token_count > 0 for non-empty source.
 */
int mirror7_parse_ir(const char *source, size_t length, mirror7_surface_ir_t *out);
int mirror7_surface_ir_validate(const mirror7_surface_ir_t *ir);

#endif
