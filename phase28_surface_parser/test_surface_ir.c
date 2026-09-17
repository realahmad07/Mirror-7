#include "surface_ir.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    const char *src = ": alpha 65 emit IF 1 ELSE 2 THEN ;";
    mirror7_surface_ir_t ir;
    if (!mirror7_parse_ir(src, strlen(src), &ir)) return 1;
    if (!mirror7_surface_ir_validate(&ir)) return 2;
    if (ir.version != MIRROR7_SURFACE_IR_VERSION || ir.token_count != 10) return 3;
    if (ir.tokens[0].kind != MIRROR7_IR_COLON || ir.tokens[0].source_pos != 0) return 4;
    if (ir.tokens[1].kind != MIRROR7_IR_NAME || strcmp(ir.tokens[1].text, "alpha") != 0) return 5;
    if (ir.tokens[2].kind != MIRROR7_IR_NUMBER || ir.tokens[2].number != 65 || ir.tokens[2].length != 0) return 6;
    if (ir.tokens[4].kind != MIRROR7_IR_IF || ir.tokens[6].kind != MIRROR7_IR_ELSE || ir.tokens[8].kind != MIRROR7_IR_THEN) return 7;
    if (ir.tokens[9].kind != MIRROR7_IR_SEMI) return 8;

    mirror7_surface_ir_t bad = ir;
    bad.version++;
    if (mirror7_surface_ir_validate(&bad)) return 9;
    bad = ir;
    bad.tokens[2].length = 1;
    if (mirror7_surface_ir_validate(&bad)) return 10;
    puts("PHASE28_2_ABI_PASS");
    return 0;
}
