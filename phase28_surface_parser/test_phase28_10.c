#include <stdio.h>
#include <stdlib.h>
#define MIRROR7_SURFACE_IR_COMPILER_BRIDGE_NO_MAIN
#include "surface_ir_compiler_bridge.c"
#undef MIRROR7_SURFACE_IR_COMPILER_BRIDGE_NO_MAIN

int main(void) {
    const char *cases[] = {
        ": p1 1 IF 65 emit THEN ;",
        ": p2 0 IF 65 emit ELSE 66 emit THEN ;",
        ": p3 1 IF 1 IF 67 emit THEN THEN ;",
        ": helper 68 emit ; : outer helper 0 IF 69 emit ELSE 70 emit THEN ;",
        ": nested 0 IF 1 IF 65 emit ELSE 66 emit THEN ELSE 1 IF 0 IF 67 emit ELSE 68 emit THEN ELSE 69 emit THEN THEN ;"
    };
    if (!compile_and_run(cases[0], "p1")) return 1;
    if (!compile_and_run(cases[1], "p2")) return 1;
    if (!compile_and_run(cases[2], "p3")) return 1;
    if (!compile_and_run(cases[3], "outer")) return 1;
    if (!compile_and_run(cases[4], "nested")) return 1;
    printf("HELDOUT_PASS\n");
    return 0;
}
