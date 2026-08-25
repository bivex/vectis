/* ======================================================================= */
/*  VECTIS NEURAL SYNTHESIZED VCPU — TARGET: INTEL CORE i3 (x86-64)       */
/*  OPTIMIZATIONS: Direct-Threading, Pinned GPRs, 64-bit 1-Cycle MBA ALU   */
/* ======================================================================= */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// Overlapping 64/32/16-bit register matrix to defeat decompiler SSA analysis
typedef union {
    uint64_t q[16];
    uint32_t d[32];
    uint16_t w[64];
    uint8_t  b[128];
} __attribute__((aligned(64))) vcpu_bank_t;

typedef struct {
    uint64_t v_pc;
    uint64_t v_key; // Rolling Feistel cryptographic state
    uint64_t v_flags;
    vcpu_bank_t bank;
} vcpu_state_t;

// Fast Direct-Threaded VCPU execution kernel
uint64_t vectis_execute_vcpu(const uint8_t *bytecode, size_t len, uint64_t initial_key) {
    vcpu_state_t state = { .v_pc = 0, .v_key = initial_key };

    // Pin hot virtual registers to host CPU GPRs (Zero RAM spill on Core i3)
    register uint64_t v_r0 asm("r12") = 0;
    register uint64_t v_r1 asm("r13") = 0;
    register uint64_t v_r2 asm("r14") = 0;
    register const uint8_t *pc asm("r15") = bytecode;

    // Direct-Threading Jump Table: Zero branch predictor stalls on Core i3 BTB
    static const void *dispatch_table[16] = {
        &&op_nop, &&op_add_mba, &&op_xor_mba, &&op_sub_mba,
        &&op_ror, &&op_rol,     &&op_load,    &&op_store,
        &&op_key_roll, &&op_loki_guard, &&op_mov, &&op_ret,
        &&op_trap, &&op_trap,   &&op_trap,    &&op_trap
    };

    #define DISPATCH() goto *dispatch_table[(*pc++) & 0x0F]

    DISPATCH();

op_add_mba:
    {
        // Stateful dynamic key mutation (1-cycle Feistel step)
        state.v_key = ((state.v_key * 0x5851F42D4C957F2DULL) ^ (state.v_key >> 17)) + 0x14057B7EF767814FULL;
        // 64-bit Non-linear MBA expansion (Resists D810 & Triton SMT Simplification)
        uint64_t a = v_r0, b = v_r1;
        v_r0 = (((a ^ ~b) + ((a & b) << 1) + 1ULL) ^ state.v_key) - state.v_key;
        DISPATCH();
    }

op_xor_mba:
    {
        uint64_t a = v_r0, b = v_r1;
        v_r0 = (a | b) - (a & b);
        DISPATCH();
    }

op_sub_mba:
    {
        uint64_t a = v_r0, b = v_r1;
        v_r0 = (a ^ ~b) + 1ULL - 2ULL * (a & ~b);
        DISPATCH();
    }

op_ror:
    {
        uint64_t a = v_r0, shift = (*pc++) & 0x3F;
        v_r0 = (a >> shift) | (a << (64 - shift));
        DISPATCH();
    }

op_rol:
    {
        uint64_t a = v_r0, shift = (*pc++) & 0x3F;
        v_r0 = (a << shift) | (a >> (64 - shift));
        DISPATCH();
    }

op_load:
    {
        uint8_t idx = (*pc++) & 0x0F;
        v_r0 = state.bank.q[idx];
        DISPATCH();
    }

op_store:
    {
        uint8_t idx = (*pc++) & 0x0F;
        state.bank.q[idx] = v_r0;
        DISPATCH();
    }

op_key_roll:
    {
        state.v_key = (state.v_key << 13) | (state.v_key >> 51);
        state.v_key ^= 0x9E3779B97F4A7C15ULL;
        DISPATCH();
    }

op_mov:
    {
        v_r1 = v_r0;
        DISPATCH();
    }

op_loki_guard:
    {
        // Loki 2-variable Algebraic Invariant (Always true, SMT solver timeout)
        uint64_t x = state.v_key;
        if (((x * x + x) & 1) != 0) { __builtin_trap(); } // Opaque Dead Branch
        DISPATCH();
    }

op_nop:
    DISPATCH();

op_trap:
    __builtin_trap();

op_ret:
    return v_r0;

}
