/* ============================================================================ */
/*  VECTIS NEURAL VCPU v4 — ENTROPY-UNIQUE BUILD                               */
/*  ISA: vISA_Vector_Arch_5BF2                                         */
/*  Seed: 349475396                                                        */
/*  ARX Rotations: 13/17                                                      */
/*  Feistel: mult=0x0000000000000899 delta=0xFE694D69                      */
/* ============================================================================ */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define VECTIS_BARRIER(x)   __asm__ volatile("" : "+r"(x))
#define VECTIS_FENCE()      __asm__ volatile("" ::: "memory")

// Per-build opaque-zero and Loki bias (Entropy seed: 349475396)
#define VECTIS_OPAQUE_0(k)  ((uint64_t)(((k) * ((k) + 1ULL)) & 1ULL))
#define VECTIS_LOKI_BIAS(k) (VECTIS_OPAQUE_0(k) * 0x2F7CE7A3421EA6F0ULL)

// Per-build dispatch map (Entropy-shuffled opcodes):
//   0x0 = NOP
//   0x1 = LOKI
//   0x2 = MOV
//   0x3 = ADD
//   0x4 = LOAD
//   0x5 = RET
//   0x6 = TRAP2
//   0x7 = TRAP
//   0x8 = TRAP3
//   0x9 = XOR
//   0xA = SUB
//   0xB = STORE
//   0xC = ARX
//   0xD = ROR
//   0xE = KEY_ROLL
//   0xF = ROL

typedef union {
    uint64_t q[16]; uint32_t d[32]; uint16_t w[64]; uint8_t b[128];
} __attribute__((aligned(64))) vcpu_bank_t;

typedef struct {
    uint64_t v_pc;
    uint64_t v_key;
    uint64_t v_flags;
    vcpu_bank_t bank;
} vcpu_state_t;

__attribute__((noinline)) static uint64_t _vcpu_init_probe(uint64_t x, uint64_t k) {
    volatile uint64_t _sink = VECTIS_LOKI_BIAS(k);
    VECTIS_BARRIER(x);
    return x ^ _sink;
}
typedef uint64_t (*_vcpu_probe_fn)(uint64_t, uint64_t);
static const _vcpu_probe_fn _vcpu_probe = _vcpu_init_probe;

uint64_t vectis_execute_vcpu(const uint8_t *bytecode, size_t len, uint64_t initial_key) {
    (void)len;
    vcpu_state_t state = { .v_pc = 0, .v_key = 0x00000000FFBC4F5EULL ^ initial_key };

    register uint64_t v_r0 asm("r12") = 0;
    register uint64_t v_r1 asm("r13") = 0;
    register const uint8_t *pc asm("r15") = bytecode;

    v_r0 = _vcpu_probe(v_r0, state.v_key);
    VECTIS_FENCE();

    static const void * const _dt[16] = {
        &&op_nop, &&op_loki, &&op_mov, &&op_add, &&op_load, &&op_ret, &&op_trap, &&op_trap, &&op_trap, &&op_xor, &&op_sub, &&op_store, &&op_arx, &&op_ror, &&op_key_roll, &&op_rol
    };
    #define DISPATCH() goto *_dt[(*pc++) & 0x0F]

    DISPATCH();

op_nop:
    DISPATCH();

op_add:
    {
        uint64_t _k1 = ((state.v_key * 0x0000000000000899ULL) ^ (state.v_key >> 17)) + 0x00000000FE694D69ULL;
        VECTIS_BARRIER(_k1);
        uint64_t _k2 = ((_k1 << 13) | (_k1 >> 51)) ^ 0x9E3779B97F4A7C15ULL;
        VECTIS_BARRIER(_k2);
        state.v_key = ((_k2 * 0xBF58476D1CE4E5B9ULL) ^ (_k2 >> 31)) + 0x94D049BB133111EBULL;
        uint64_t _a = v_r0, _b = v_r1;
        uint64_t _t0 = (_a | _b); VECTIS_BARRIER(_t0);
        uint64_t _t1 = (_a ^ _b); VECTIS_BARRIER(_t1);
        uint64_t _t2 = 2ULL * _t0; VECTIS_BARRIER(_t2);
        uint64_t _bias = VECTIS_LOKI_BIAS(state.v_key); VECTIS_BARRIER(_bias);
        v_r0 = _t2 - _t1 + _bias;
        DISPATCH();
    }

op_xor:
    {
        uint64_t _a = v_r0, _b = v_r1;
        uint64_t _s0 = (_a | _b); VECTIS_BARRIER(_s0);
        uint64_t _s1 = (_a & ~_b); VECTIS_BARRIER(_s1);
        uint64_t _s2 = _s0 + _s1; VECTIS_BARRIER(_s2);
        uint64_t _km = state.v_key; VECTIS_BARRIER(_km);
        uint64_t _s3 = _s2 + _km; VECTIS_BARRIER(_s3);
        v_r0 = _s3 - _a - _km;
        DISPATCH();
    }

op_sub:
    {
        uint64_t _a = v_r0, _b = v_r1;
        uint64_t _nb = ~_b; VECTIS_BARRIER(_nb);
        uint64_t _t0 = _a + _nb; VECTIS_BARRIER(_t0);
        v_r0 = _t0 + 1ULL;
        DISPATCH();
    }

op_ror:
    {
        uint64_t _a = v_r0, _sh = (*pc++) & 0x3F;
        v_r0 = (_a >> _sh) | (_a << (64 - _sh));
        DISPATCH();
    }

op_rol:
    {
        uint64_t _a = v_r0, _sh = (*pc++) & 0x3F;
        v_r0 = (_a << _sh) | (_a >> (64 - _sh));
        DISPATCH();
    }

op_load:
    {
        uint8_t _i = (*pc++) & 0x0F;
        v_r0 = state.bank.q[_i];
        DISPATCH();
    }

op_store:
    {
        uint8_t _i = (*pc++) & 0x0F;
        state.bank.q[_i] = v_r0;
        DISPATCH();
    }

op_key_roll:
    {
        uint64_t _k = state.v_key;
        _k = ((_k << 13) | (_k >> 51)) ^ 0x9E3779B97F4A7C15ULL;
        VECTIS_BARRIER(_k);
        _k = ((_k * 0xBF58476D1CE4E5B9ULL) ^ (_k >> 31)) + 0x94D049BB133111EBULL;
        VECTIS_BARRIER(_k);
        _k = ((_k ^ (_k >> 33)) * 0xFF51AFD7ED558CCDULL) ^ (_k >> 33);
        state.v_key = _k;
        DISPATCH();
    }

op_loki:
    {
        uint64_t _x = state.v_key; VECTIS_BARRIER(_x);
        if (((_x * (_x + 1ULL)) & 1ULL) != 0) { __builtin_trap(); }
        DISPATCH();
    }

op_arx:
    {
        uint64_t _k = state.v_key;
        uint64_t _t = v_r0 + _k; VECTIS_BARRIER(_t);
        _t = ((_t << 13) | (_t >> 51)) ^ _k; VECTIS_BARRIER(_t);
        _t = _t + v_r1; VECTIS_BARRIER(_t);
        v_r0 = ((_t << 17) | (_t >> 47)) ^ v_r1;
        DISPATCH();
    }

op_mov:
    {
        v_r1 = v_r0;
        DISPATCH();
    }

op_ret:
        return v_r0 ^ VECTIS_LOKI_BIAS(state.v_key);

op_trap:
    __builtin_trap();

}

#include <stdio.h>
int main(void) {
    uint8_t prog[] = { 0x0, 0x1, 0x5 };
    uint64_t r = vectis_execute_vcpu(prog, sizeof(prog), 0x1337BABE00000000ULL);
    printf("[+] 0x%016llx\n", (unsigned long long)r);
    return 0;
}
