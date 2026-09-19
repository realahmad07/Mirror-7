#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Keep the existing nucleus entry point out of this test translation unit. */
#define MIRROR7_NO_MAIN
#define main mirror7_nucleus_main
#include "../phase25_26/nucleus.c"
#include "parser.c"
#undef main

#include "surface_ir.h"

typedef struct {
    size_t if_patch;
    size_t else_patch;
    int has_else;
} mirror7_flow_t;

static int patch_u16_at(dict_t *d, size_t at, uint16_t value) {
    if (at + 1 >= d->n) return 0;
    d->code[at] = (uint8_t)value;
    d->code[at + 1] = (uint8_t)(value >> 8);
    return 1;
}

static int emit_branch_placeholder(dict_t *d, const char *op, size_t *patch) {
    size_t before = d->n;
    if (!compile_token(d, op)) return 0;
    if (d->n != before + 3) return 0;
    *patch = before + 1;
    return 1;
}

/*
 * Phase 28.3 boundary: consume the parser's structured IR directly and
 * lower it into the existing MIRR compiler dictionary/code representation.
 * The Phase 27 raw-source compiler path remains untouched.
 */
int mirror7_compile_surface_ir(const mirror7_surface_ir_t *ir, dict_t *d) {
    if (!ir || !d || !mirror7_surface_ir_validate(ir)) return 0;

    memset(d, 0, sizeof(*d));
    if (!init_primitives(d)) return 0;

    size_t i = 0;
    while (i < ir->token_count) {
        const mirror7_ir_token_t *t = &ir->tokens[i++];
        if (t->kind != MIRROR7_IR_COLON || i >= ir->token_count) return 0;

        const mirror7_ir_token_t *name = &ir->tokens[i++];
        if (name->kind != MIRROR7_IR_NAME || name->length == 0 || name->length > MAX_NAME) return 0;
        if (find_word(d, name->text) >= 0 || d->nw >= MAX_WORDS) return 0;

        size_t word_id = d->nw;
        uint32_t start = (uint32_t)d->n;
        strcpy(d->words[word_id].name, name->text);
        d->words[word_id].off = start;
        d->words[word_id].len = 0;
        d->nw++;

        mirror7_flow_t flow[1024];
        size_t fp = 0;
        int ended = 0;

        while (i < ir->token_count) {
            t = &ir->tokens[i++];

            if (t->kind == MIRROR7_IR_SEMI) {
                if (fp != 0 || !emitb(d, OP_RET)) return 0;
                d->words[word_id].len = (uint32_t)(d->n - start);
                ended = 1;
                break;
            }

            if (t->kind == MIRROR7_IR_IF) {
                if (fp >= 1024 || !emit_branch_placeholder(d, "0branch", &flow[fp].if_patch)) return 0;
                flow[fp].else_patch = 0;
                flow[fp].has_else = 0;
                fp++;
                continue;
            }

            if (t->kind == MIRROR7_IR_ELSE) {
                if (fp == 0 || flow[fp - 1].has_else) return 0;
                mirror7_flow_t *f = &flow[fp - 1];
                if (!emit_branch_placeholder(d, "branch", &f->else_patch)) return 0;
                if (!patch_u16_at(d, f->if_patch, (uint16_t)d->n)) return 0;
                f->has_else = 1;
                continue;
            }

            if (t->kind == MIRROR7_IR_THEN) {
                if (fp == 0) return 0;
                mirror7_flow_t f = flow[--fp];
                if (!patch_u16_at(d, f.has_else ? f.else_patch : f.if_patch, (uint16_t)d->n)) return 0;
                continue;
            }

            if (t->kind == MIRROR7_IR_NUMBER) {
                if (t->number > 255) return 0;
                char num[6];
                snprintf(num, sizeof(num), "%u", (unsigned)t->number);
                if (!compile_token(d, num)) return 0;
                continue;
            }

            if (t->kind == MIRROR7_IR_NAME) {
                if (!compile_token(d, t->text)) return 0;
                continue;
            }

            return 0;
        }

        if (!ended) return 0;
    }

    for (size_t p = 0; p < d->npending; ++p) {
        int id = find_word(d, d->pending[p].name);
        if (id < 0 || d->pending[p].pos + 1 >= d->n) return 0;
        d->code[d->pending[p].pos] = (uint8_t)id;
        d->code[d->pending[p].pos + 1] = (uint8_t)(id >> 8);
    }

    return 1;
}

static int compile_and_run(const char *source, const char *word) {
    mirror7_surface_ir_t ir;
    dict_t d;
    if (!mirror7_parse_ir(source, strlen(source), &ir)) return 0;
    if (!mirror7_compile_surface_ir(&ir, &d)) return 0;
    int id = find_word(&d, word);
    if (id < 0) return 0;
    return run_word(&d, &d, id, (uint8_t *)source, strlen(source));
}

#ifndef MIRROR7_SURFACE_IR_COMPILER_BRIDGE_NO_MAIN
int main(void) {
    const char *cases[] = {
        ": alpha 1 IF 65 emit ELSE 66 emit THEN ;",
        ": alpha 0 IF 65 emit ELSE 66 emit THEN ;",
        ": alpha 1 IF 1 IF 65 emit THEN ELSE 66 emit THEN ;",
        ": alpha 1 IF 65 emit THEN ;",
        ": alpha helper ; : helper 65 emit ;"
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        if (!compile_and_run(cases[i], "alpha")) {
            fprintf(stderr, "PHASE28_3_FAIL case=%zu\n", i);
            return 1;
        }
    }
    return 0;
}
#endif
