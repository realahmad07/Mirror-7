#include <stdio.h>
#include <stdlib.h>

#define MIRROR7_SURFACE_IR_COMPILER_BRIDGE_NO_MAIN
#include "surface_ir_compiler_bridge.c"
#undef MIRROR7_SURFACE_IR_COMPILER_BRIDGE_NO_MAIN

int main(int argc, char **argv) {
    static const char *cases[] = {
        ": alpha 65 emit ;",
        ": alpha 1 IF 65 emit ELSE 66 emit THEN ;",
        ": alpha 0 IF 65 emit ELSE 66 emit THEN ;",
        ": alpha 1 IF 1 IF 65 emit THEN ELSE 66 emit THEN ;",
        ": alpha helper ; : helper 65 emit ;",
        ": helper 65 emit ; : alpha helper ;",
        ": helper 65 emit ; : alpha helper helper ;",
        ": a 65 emit ; : b a ; : alpha b ;"
    };
    if (argc != 2) return 2;
    int n = atoi(argv[1]);
    if (n < 0 || (size_t)n >= sizeof(cases) / sizeof(cases[0])) return 2;
    if (!compile_and_run(cases[n], "alpha")) return 1;
    return 0;
}
