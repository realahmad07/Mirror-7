#include <stdio.h>
#include <stdlib.h>

#define MIRROR7_SURFACE_IR_COMPILER_BRIDGE_NO_MAIN
#include "surface_ir_compiler_bridge.c"
#undef MIRROR7_SURFACE_IR_COMPILER_BRIDGE_NO_MAIN

static const char *cases[] = {
    ": alpha 1 IF 1 IF 65 emit ELSE 66 emit THEN ELSE 67 emit THEN ;",
    ": alpha 1 IF 0 IF 65 emit ELSE 66 emit THEN ELSE 67 emit THEN ;",
    ": alpha 0 IF 1 IF 65 emit ELSE 66 emit THEN ELSE 67 emit THEN ;",
    ": alpha 0 IF 0 IF 65 emit ELSE 66 emit THEN ELSE 67 emit THEN ;",
    ": alpha 1 IF 1 IF 1 IF 65 emit ELSE 66 emit THEN ELSE 67 emit THEN ELSE 68 emit THEN ;",
    ": alpha 1 IF 0 IF 1 IF 65 emit ELSE 66 emit THEN ELSE 67 emit THEN ELSE 68 emit THEN ;",
    ": alpha 0 IF 1 IF 1 IF 65 emit ELSE 66 emit THEN ELSE 67 emit THEN ELSE 68 emit THEN ;",
    ": alpha 1 IF 65 emit ELSE 0 IF 66 emit ELSE 67 emit THEN THEN ;",
    ": alpha 0 IF 65 emit ELSE 1 IF 66 emit ELSE 67 emit THEN THEN ;",
    ": alpha 1 IF 0 IF 65 emit ELSE 66 emit THEN ELSE 1 IF 67 emit ELSE 68 emit THEN THEN ;"
};

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    int n = atoi(argv[1]);
    if (n < 0 || (size_t)n >= sizeof(cases) / sizeof(cases[0])) return 2;
    if (!compile_and_run(cases[n], "alpha")) return 1;
    return 0;
}
