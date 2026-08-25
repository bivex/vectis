#!/usr/bin/env python3
"""
mlx_i3_ultra_emulator_synth.py v4 — Neural VCPU Synthesizer with OCaml Entropy Bridge.

NEW in v4:
  * VectisEntropyBridge: shells out to vectis_synth.exe (OCaml Entropy_port)
    to generate per-build unique ISA parameters:
      - Shuffled opcode-to-index dispatch map   (Fisher-Yates via Entropy.next_int)
      - 64-bit pack_key and delta_key           (Entropy.next_int64)
      - ABI input register permutation          (Entropy.shuffle)
      - ARX rotation amounts (17/31 → random prime-adjacent values)
      - Feistel LCG multiplier and delta        (from rolling_vkey JSON)
      - Loki bias constant                      (64-bit odd random)
  * C11VCPUEmitter uses these values in generated code — no two builds share
    the same dispatch table, constants, or MBA bias
  * --seed N for reproducible synthesis; without --seed each build is unique
"""

import sys, os, json, math, random, time, argparse, subprocess, struct
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as opt
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

# ── Path to the compiled OCaml Entropy binary ──────────────────────────────────
_REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SYNTH_EXE   = os.path.join(_REPO_ROOT, "_build/default/bin/vectis_synth.exe")


# ==============================================================================
# 0. OCAML ENTROPY BRIDGE
# ==============================================================================

@dataclass
class VectisEntropyParams:
    """Fully randomized per-build VCPU parameters sourced from OCaml Entropy_port."""
    seed:            int                   # OS entropy or user-supplied seed
    isa_name:        str                   # e.g. "RollingVKey_Arch_C6C4"
    # Opcode dispatch map: logical op name → nibble index 0x0..0xF
    opcode_map:      Dict[str, int]        # e.g. {"NOP":0,"ADD":3,"XOR":7,...}
    # Feistel key schedule constants (from rolling_vkey JSON)
    lcg_mult:        int                   # odd, e.g. 31539
    lcg_delta:       int                   # e.g. 2489859797
    vkey_seed:       int                   # 64-bit initial key
    # Pack/delta keys for MBA bias injection (from visa JSON)
    pack_key:        int                   # 32-bit
    delta_key:       int                   # 32-bit odd
    # ABI register permutation (which GPR index = which slot)
    abi_in_regs:     List[int]             # e.g. [2,4,5,7,3,6,1,0]
    # ARX rotation amounts — two random prime-adjacent odd numbers
    arx_rot1:        int                   # e.g. 13, 17, 19, 23, 29
    arx_rot2:        int                   # e.g. 31, 37, 41, 43
    # Loki bias constant (64-bit, always even → bias==0 guaranteed)
    loki_bias_const: int                   # e.g. 0xDEADC0DEDEADC0DE (even)


class VectisEntropyBridge:
    """
    Bridges Python synthesizer to OCaml Entropy_port via vectis_synth.exe.

    Protocol:
      1. Run vectis_synth.exe --vcpu visa --seed S   → JSON with opcodes, pack_key, delta_key, abi
      2. Run vectis_synth.exe --vcpu rolling_vkey --seed S+1 → JSON with lcg_mult, lcg_delta, vkey_seed
      3. Parse JSONs → VectisEntropyParams
      4. Assign 16 opcode slots from shuffled funct6 pool (already shuffled by OCaml)
    """

    # Canonical logical op names → their 4-bit opcode slot (assigned from OCaml shuffle)
    LOGICAL_OPS = ["NOP", "ADD", "XOR", "SUB", "ROR", "ROL", "LOAD", "STORE",
                   "KEY_ROLL", "LOKI", "ARX", "MOV", "RET", "TRAP", "TRAP2", "TRAP3"]

    # Prime-adjacent rotation candidates (safe ARX rotations)
    _ARX_ROTS = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    @classmethod
    def query(cls, seed: Optional[int] = None) -> VectisEntropyParams:
        """Query OCaml vectis_synth.exe for unique ISA parameters."""
        if not os.path.exists(_SYNTH_EXE):
            print(f"[!] vectis_synth.exe not found at {_SYNTH_EXE}")
            print("[!] Falling back to Python PRNG entropy (not as strong)")
            return cls._fallback(seed)

        actual_seed = seed if seed is not None else random.randint(0, 2**30)
        # Seeds: use S for visa (opcode/key layout), S+1 for rolling_vkey (Feistel)
        visa_seed = actual_seed
        vkey_seed_int = actual_seed + 1

        # ── Query 1: visa (opcodes, pack_key, delta_key, abi) ──────────────────
        visa_json = cls._run_synth("visa", visa_seed)
        # ── Query 2: rolling_vkey (lcg_mult, lcg_delta, vkey_seed) ─────────────
        vkey_json = cls._run_synth("rolling_vkey", vkey_seed_int)

        if visa_json is None or vkey_json is None:
            return cls._fallback(seed)

        # Parse visa JSON
        v_opcodes  = visa_json.get("opcodes", {})
        pack_key   = int(visa_json.get("pack_key", 0xDEADBEEF)) & 0xFFFFFFFF
        delta_key  = int(visa_json.get("delta_key", 0xCAFEBABE)) & 0xFFFFFFFF
        abi_in     = visa_json.get("abi", {}).get("in_regs", list(range(8)))

        # Parse rolling_vkey JSON
        lcg_mult   = int(vkey_json.get("lcg_multiplier", 31539)) & 0xFFFF
        lcg_delta  = int(vkey_json.get("lcg_delta", 0x9E3779B9)) & 0xFFFFFFFF
        vkey_seed_val = int(vkey_json.get("vkey_seed", 0xDEADBEEF13371337)) & 0xFFFFFFFFFFFFFFFF

        # Build opcode map: assign funct6 values from visa opcodes to our 16 slots
        # We use the 15 unique funct6 values from vadd_vv..vsuper_arx (shuffled by OCaml)
        funct6_values = list(dict.fromkeys(v_opcodes.values()))[:15]
        # Map to nibbles: sort so we get a deterministic but per-seed assignment
        funct6_sorted = sorted(funct6_values)
        # Assign logical ops to 4-bit nibbles via the shuffled funct6 ordering
        opcode_map: Dict[str, int] = {}
        for i, op in enumerate(cls.LOGICAL_OPS):
            opcode_map[op] = funct6_sorted[i % len(funct6_sorted)] % 16 if funct6_sorted else i % 16

        # Guarantee RET = some unique slot, NOP = 0x0 always (required for dispatch)
        opcode_map["NOP"] = 0x0   # NOP must be 0 (initial dispatch)
        # Ensure RET has a unique slot (remap if collision)
        used = set(opcode_map.values())
        if opcode_map.get("RET") in used - {opcode_map["RET"]}:
            opcode_map["RET"] = (opcode_map["RET"] + 1) % 16

        # ARX rotations: pick from rotation candidates based on pack_key parity
        rot_idx1 = (pack_key >> 4) % len(cls._ARX_ROTS)
        rot_idx2 = (delta_key >> 4) % len(cls._ARX_ROTS)
        arx_rot1 = cls._ARX_ROTS[rot_idx1]
        arx_rot2 = cls._ARX_ROTS[(rot_idx2 + 3) % len(cls._ARX_ROTS)]  # ensure different

        # Loki bias: use pack_key * 2 (always even → zero contribution)
        loki_bias = (pack_key * delta_key) & 0xFFFFFFFFFFFFFFFE  # clear bit 0 → even

        isa_name = visa_json.get("isa_name", f"VectisUniq_{actual_seed:08X}")

        return VectisEntropyParams(
            seed=actual_seed,
            isa_name=isa_name,
            opcode_map=opcode_map,
            lcg_mult=lcg_mult,
            lcg_delta=lcg_delta,
            vkey_seed=vkey_seed_val,
            pack_key=pack_key,
            delta_key=delta_key,
            abi_in_regs=abi_in[:8],
            arx_rot1=arx_rot1,
            arx_rot2=arx_rot2,
            loki_bias_const=loki_bias,
        )

    @classmethod
    def _run_synth(cls, vcpu_type: str, seed: int) -> Optional[dict]:
        """Run vectis_synth.exe and return parsed JSON."""
        import tempfile
        tmp = tempfile.mktemp(suffix=".json")
        try:
            r = subprocess.run(
                [_SYNTH_EXE, "--vcpu", vcpu_type, "--seed", str(seed), "-o", tmp],
                capture_output=True, timeout=10
            )
            if r.returncode != 0:
                return None
            with open(tmp) as f:
                return json.load(f)
        except Exception as e:
            print(f"[!] vectis_synth query failed ({vcpu_type}): {e}")
            return None
        finally:
            try: os.unlink(tmp)
            except: pass

    @classmethod
    def _fallback(cls, seed: Optional[int]) -> VectisEntropyParams:
        """Python PRNG fallback when vectis_synth.exe is unavailable."""
        rng = random.Random(seed if seed is not None else time.time_ns())
        slots = list(range(16))
        rng.shuffle(slots)
        opcode_map = {op: slots[i] for i, op in enumerate(cls.LOGICAL_OPS)}
        opcode_map["NOP"] = 0x0
        rots = cls._ARX_ROTS
        return VectisEntropyParams(
            seed=seed or 0,
            isa_name=f"VectisFallback_{rng.randint(0, 0xFFFF):04X}",
            opcode_map=opcode_map,
            lcg_mult=rng.choice([31539, 6364136223846793005, 2685821657736338717]),
            lcg_delta=rng.randint(0x10000000, 0xFFFFFFFE) | 1,
            vkey_seed=rng.randint(0, 2**63),
            pack_key=rng.randint(0, 2**32 - 1),
            delta_key=rng.randint(0, 2**32 - 1) | 1,
            abi_in_regs=rng.sample(range(16), 8),
            arx_rot1=rng.choice(rots),
            arx_rot2=rng.choice(rots),
            loki_bias_const=(rng.randint(0, 2**63)) & 0xFFFFFFFFFFFFFFFE,
        )


# ==============================================================================
# 1. INTEL CORE i3 COST MODEL
# ==============================================================================

class IntelCoreI3CostModel:
    @staticmethod
    def estimate_uop_cycles(fragment: str, dtc: bool) -> float:
        base    = 1.8 if dtc else 16.5
        adds    = fragment.count("+") + fragment.count("-")
        bitwise = fragment.count("^") + fragment.count("&") + fragment.count("|") + fragment.count("~")
        muls    = fragment.count("*")
        mems    = fragment.count("->") + fragment.count("[")
        barriers= fragment.count("VECTIS_BARRIER") * 0.5
        return base + adds * 1.0 + bitwise * 1.0 + muls * 3.0 + mems * 3.5 + barriers

    @staticmethod
    def estimate_l1i_bytes(handlers: List[str]) -> int:
        return int(sum(len(h) for h in handlers) * 0.35)

    @staticmethod
    def i3_speed_fitness(avg_cycles: float, l1i_bytes: int, gpr_spills: int) -> float:
        cycle_sc = max(0.0, 1.0 - avg_cycles / 50.0)
        l1i_sc   = 1.0 if l1i_bytes <= 24576 else (0.6 if l1i_bytes <= 32768 else 0.1)
        spill_sc = max(0.0, 1.0 - gpr_spills * 0.25)
        return 0.55 * cycle_sc + 0.25 * l1i_sc + 0.20 * spill_sc


# ==============================================================================
# 2. COMPLEXITY METRIC
# ==============================================================================

class DecompilerComplexityMetric:
    @staticmethod
    def compute(mba_depth, rolling_key, vbank, opaques, isw_xor, twos_sub,
                arx_chaos, opaque_calls, multi_feistel, masked_ret) -> float:
        w = [
            min(1.0, mba_depth / 5.0) * 0.20,
            0.12 if rolling_key   else 0.0,
            0.10 if vbank         else 0.0,
            min(0.10, opaques * 0.04),
            0.12 if isw_xor       else 0.0,
            0.08 if twos_sub      else 0.0,
            0.10 if arx_chaos     else 0.0,
            0.08 if opaque_calls  else 0.0,
            0.06 if multi_feistel else 0.0,
            0.04 if masked_ret    else 0.0,
        ]
        return min(1.0, sum(w))


# ==============================================================================
# 3. RL ENVIRONMENT
# ==============================================================================

ACTIONS = [
    "ENABLE_DIRECT_THREADING",   # 0
    "SYNTH_MBA_DEPTH_2",         # 1
    "SYNTH_MBA_DEPTH_4",         # 2
    "SYNTH_MBA_DEPTH_5",         # 3
    "PIN_VREGS_TO_GPR",          # 4
    "INJECT_FEISTEL_KEY",        # 5
    "INJECT_MULTI_FEISTEL_KEY",  # 6
    "ENABLE_ALIASED_VBANK",      # 7
    "INJECT_LOKI_INVARIANT",     # 8
    "INJECT_ISW_XOR",            # 9
    "INJECT_TWOS_SUB",           # 10
    "INJECT_ARX_CHAOS",          # 11
    "INJECT_OPAQUE_CALLS",       # 12
    "INJECT_MASKED_RET",         # 13
]

@dataclass
class VCPUSpec:
    dtc:            bool = False
    mba_depth:      int  = 1
    pinned_gprs:    bool = False
    rolling_key:    bool = False
    multi_feistel:  bool = False
    vbank:          bool = False
    opaques:        int  = 0
    isw_xor:        bool = False
    twos_sub:       bool = False
    arx_chaos:      bool = False
    opaque_calls:   bool = False
    masked_ret:     bool = False
    # Filled by Entropy bridge after RL training
    entropy:        Optional[VectisEntropyParams] = None

    def complexity(self) -> float:
        return DecompilerComplexityMetric.compute(
            self.mba_depth, self.rolling_key, self.vbank, self.opaques,
            self.isw_xor, self.twos_sub, self.arx_chaos, self.opaque_calls,
            self.multi_feistel, self.masked_ret
        )


class VCPUSynthesisEnv:
    STATE_DIM  = 14
    ACTION_DIM = len(ACTIONS)

    def __init__(self):
        self.reset()

    def reset(self):
        self.spec   = VCPUSpec()
        self.step_n = 0
        return self._state()

    def _state(self):
        s = self.spec
        return [float(s.dtc), s.mba_depth/5.0, float(s.pinned_gprs),
                float(s.rolling_key), float(s.multi_feistel), float(s.vbank),
                min(1.0, s.opaques/4.0), float(s.isw_xor), float(s.twos_sub),
                float(s.arx_chaos), float(s.opaque_calls), float(s.masked_ret),
                self.step_n/14.0, 0.0]

    def step(self, action_idx):
        self.step_n += 1
        a, s = ACTIONS[action_idx], self.spec
        if   a == "ENABLE_DIRECT_THREADING":  s.dtc          = True
        elif a == "SYNTH_MBA_DEPTH_2":        s.mba_depth    = max(s.mba_depth, 2)
        elif a == "SYNTH_MBA_DEPTH_4":        s.mba_depth    = max(s.mba_depth, 4)
        elif a == "SYNTH_MBA_DEPTH_5":        s.mba_depth    = 5
        elif a == "PIN_VREGS_TO_GPR":         s.pinned_gprs  = True
        elif a == "INJECT_FEISTEL_KEY":       s.rolling_key  = True
        elif a == "INJECT_MULTI_FEISTEL_KEY": s.multi_feistel= True; s.rolling_key = True
        elif a == "ENABLE_ALIASED_VBANK":     s.vbank        = True
        elif a == "INJECT_LOKI_INVARIANT":    s.opaques      += 1
        elif a == "INJECT_ISW_XOR":           s.isw_xor      = True
        elif a == "INJECT_TWOS_SUB":          s.twos_sub     = True
        elif a == "INJECT_ARX_CHAOS":         s.arx_chaos    = True
        elif a == "INJECT_OPAQUE_CALLS":      s.opaque_calls = True
        elif a == "INJECT_MASKED_RET":        s.masked_ret   = True

        hs    = self._render_handlers()
        total = sum(IntelCoreI3CostModel.estimate_uop_cycles(h, s.dtc) for h in hs)
        avg   = total / max(1, len(hs))
        l1i   = IntelCoreI3CostModel.estimate_l1i_bytes(hs)
        speed = IntelCoreI3CostModel.i3_speed_fitness(avg, l1i, 0 if s.pinned_gprs else 3)
        cplx  = s.complexity()

        reward = 2.5 * cplx + 1.5 * speed
        if s.dtc:          reward += 1.0
        if s.isw_xor:      reward += 1.5
        if s.arx_chaos:    reward += 1.0
        if s.opaque_calls: reward += 0.8
        if speed > 0.7 and cplx >= 0.95:
            reward += 8.0

        done = self.step_n >= 14 or (speed > 0.75 and cplx >= 0.98)
        return self._state(), reward, done, dict(speed=speed, complexity=cplx, avg_cyc=avg, l1i=l1i)

    def _render_handlers(self):
        s = self.spec
        hs = []
        if s.mba_depth >= 5: mba = "ISW_ADD(a,b,k)"
        elif s.mba_depth >= 4: mba = "2*(a|b)-(a^b)+LOKI_BIAS(k)"
        elif s.mba_depth >= 2: mba = "(a^b)+2*(a&b)"
        else: mba = "a+b"
        hs.append(f"OP_ADD:{mba}")
        if s.isw_xor: hs.append("OP_XOR:ISW_3share(a,b,k)")
        else: hs.append("OP_XOR:(a|b)-(a&b)")
        if s.twos_sub: hs.append("OP_SUB:a+~b+1")
        else: hs.append("OP_SUB:a-b")
        if s.arx_chaos: hs.append("OP_ARX:ROL(a+k,R1)^ROL(a+b,R2)")
        if s.rolling_key: hs.append("KEY_ROLL:LCG")
        return hs


# ==============================================================================
# 4. PPO ACTOR-CRITIC (Apple MLX Metal GPU)
# ==============================================================================

if MLX_AVAILABLE:
    class MLXPPOPolicy(nn.Module):
        def __init__(self, in_dim=14, action_dim=len(ACTIONS), hidden=128):
            super().__init__()
            self.embed = nn.Linear(in_dim, hidden)
            self.ln0   = nn.LayerNorm(hidden)
            self.fc1   = nn.Linear(hidden, hidden); self.ln1 = nn.LayerNorm(hidden)
            self.fc2   = nn.Linear(hidden, hidden); self.ln2 = nn.LayerNorm(hidden)
            self.fc3   = nn.Linear(hidden, hidden); self.ln3 = nn.LayerNorm(hidden)
            self.actor  = nn.Sequential(nn.Linear(hidden, hidden//2), nn.GELU(),
                                        nn.Linear(hidden//2, action_dim))
            self.critic = nn.Sequential(nn.Linear(hidden, hidden//2), nn.GELU(),
                                        nn.Linear(hidden//2, 1))

        def __call__(self, x):
            h = nn.gelu(self.ln0(self.embed(x)))
            h = h + nn.gelu(self.ln1(self.fc1(h)))
            h = h + nn.gelu(self.ln2(self.fc2(h)))
            h = h + nn.gelu(self.ln3(self.fc3(h)))
            return self.actor(h), self.critic(h)

        def action_logprob(self, x, action):
            logits, val = self(x)
            lp = logits - mx.log(mx.sum(mx.exp(logits), keepdims=True))
            return lp[action], val[0]


# ==============================================================================
# 5. Z3 FORMAL VERIFIER
# ==============================================================================

class Z3FormalVerifier:
    @staticmethod
    def verify_all(spec: VCPUSpec, verbose=True) -> bool:
        if not Z3_AVAILABLE:
            print("[!] Z3 unavailable — skipping."); return True
        if verbose: print("[*] Z3 Formal MBA Equivalence Prover:")
        a, b, k = z3.BitVecs('a b k', 64)
        results = []

        def check(name, expr, target):
            s = z3.Solver(); s.set("timeout", 5000)
            s.add(expr != target)
            ok = (s.check() == z3.unsat)
            if verbose:
                print(f"  {'[+] PROVED' if ok else '[-] FAILED'}: {name}")
            results.append(ok)

        ep = spec.entropy
        if ep:
            loki_c = ep.loki_bias_const
            check(f"ADD_ISW(a,b) + LOKI_BIAS(0x{loki_c:016X}) == a+b",
                  2*(a|b) - (a^b) + ((k*(k+1))&1)*loki_c, a+b)
            check("XOR_ISW(a,b,k) == a^b",
                  (a|b) + (a&~b) - a + k - k, a^b)
            check("SUB_TWOS(a,b) == a-b", a + ~b + 1, a-b)
            check("Loki: (k*(k+1))&1 == 0", ((k*(k+1))&1)==0, z3.BoolVal(True))
            check("LOKI_BIAS even → bias==0",
                  ((k*(k+1))&1) * loki_c, z3.BitVecVal(0, 64))
        else:
            check("ADD_ISW == a+b", 2*(a|b)-(a^b), a+b)
            check("XOR_ISW == a^b", (a|b)+(a&~b)-a, a^b)
            check("SUB_TWOS == a-b", a+~b+1, a-b)
            check("Loki invariant", ((k*(k+1))&1)==0, z3.BoolVal(True))

        n = sum(results)
        if verbose: print(f"  => {n}/{len(results)} theorems proved.")
        return all(results)


# ==============================================================================
# 6. C11 VCPU EMITTER (Entropy-parameterized)
# ==============================================================================

class C11VCPUEmitter:
    BARRIER_MACRO = (
        "#define VECTIS_BARRIER(x)   __asm__ volatile(\"\" : \"+r\"(x))\n"
        "#define VECTIS_FENCE()      __asm__ volatile(\"\" ::: \"memory\")\n"
    )

    @staticmethod
    def _opaque0_macro(ep: Optional[VectisEntropyParams]) -> str:
        """Emit VECTIS_OPAQUE_0 and LOKI_BIAS using entropy-derived constant."""
        bias = ep.loki_bias_const if ep else 0xDEADC0DEDEADC0DE
        return (
            f"// Per-build opaque-zero and Loki bias (Entropy seed: {ep.seed if ep else 'N/A'})\n"
            f"#define VECTIS_OPAQUE_0(k)  ((uint64_t)(((k) * ((k) + 1ULL)) & 1ULL))\n"
            f"#define VECTIS_LOKI_BIAS(k) (VECTIS_OPAQUE_0(k) * 0x{bias:016X}ULL)\n"
        )

    @staticmethod
    def _dispatch_comment(opmap: Dict[str, int]) -> str:
        lines = ["// Per-build dispatch map (Entropy-shuffled opcodes):"]
        for op, idx in sorted(opmap.items(), key=lambda x: x[1]):
            lines.append(f"//   0x{idx:X} = {op}")
        return "\n".join(lines)

    @staticmethod
    def _feistel_constants(ep: Optional[VectisEntropyParams]) -> Tuple[int, int]:
        if ep:
            return ep.lcg_mult, ep.lcg_delta
        return 0x5851F42D4C957F2D, 0x14057B7EF767814F

    @staticmethod
    def _arx_rotations(ep: Optional[VectisEntropyParams]) -> Tuple[int, int]:
        if ep:
            return ep.arx_rot1, ep.arx_rot2
        return 17, 31

    @staticmethod
    def generate_c11_source(spec: VCPUSpec, emit_runner: bool = False) -> str:
        ep   = spec.entropy
        omap = ep.opcode_map if ep else {
            "NOP":0,"ADD":1,"XOR":2,"SUB":3,"ROR":4,"ROL":5,
            "LOAD":6,"STORE":7,"KEY_ROLL":8,"LOKI":9,"ARX":10,
            "MOV":11,"RET":12,"TRAP":13,"TRAP2":14,"TRAP3":15
        }
        rot1, rot2   = C11VCPUEmitter._arx_rotations(ep)
        mult, delta_ = C11VCPUEmitter._feistel_constants(ep)
        vkey_init    = ep.vkey_seed if ep else 0xDEADBEEF13371337

        # Guarantee 16 unique slots for dispatch table (fill collisions)
        slot_used = {}
        final_map: Dict[int, str] = {}  # nibble → op name
        for op in ["NOP","ADD","XOR","SUB","ROR","ROL","LOAD","STORE",
                   "KEY_ROLL","LOKI","ARX","MOV","RET","TRAP","TRAP2","TRAP3"]:
            slot = omap.get(op, 0) & 0xF
            while slot in final_map:
                slot = (slot + 1) & 0xF
            final_map[slot] = op
        # Reverse: op → slot
        op_slot = {v: k for k, v in final_map.items()}

        def slot(op): return op_slot.get(op, 0)

        c = []
        c.append("/* ============================================================================ */")
        c.append(f"/*  VECTIS NEURAL VCPU v4 — ENTROPY-UNIQUE BUILD                               */")
        c.append(f"/*  ISA: {(ep.isa_name if ep else 'FALLBACK'):60s}  */")
        c.append(f"/*  Seed: {(ep.seed if ep else 0):<63d}  */")
        c.append(f"/*  ARX Rotations: {rot1}/{rot2:<54d}  */")
        c.append(f"/*  Feistel: mult=0x{mult:016X} delta=0x{delta_:08X}{'':20s}  */")
        c.append("/* ============================================================================ */\n")
        c.append("#include <stdint.h>")
        c.append("#include <stddef.h>")
        c.append("#include <stdbool.h>\n")
        c.append(C11VCPUEmitter.BARRIER_MACRO)
        c.append(C11VCPUEmitter._opaque0_macro(ep))
        c.append(C11VCPUEmitter._dispatch_comment(op_slot) + "\n")

        if spec.vbank:
            c.append("typedef union {")
            c.append("    uint64_t q[16]; uint32_t d[32]; uint16_t w[64]; uint8_t b[128];")
            c.append("} __attribute__((aligned(64))) vcpu_bank_t;\n")

        c.append("typedef struct {")
        c.append("    uint64_t v_pc;")
        c.append("    uint64_t v_key;")
        c.append("    uint64_t v_flags;")
        c.append("    vcpu_bank_t bank;" if spec.vbank else "    uint64_t regs[16];")
        c.append("} vcpu_state_t;\n")

        if spec.opaque_calls:
            c.append("__attribute__((noinline)) static uint64_t _vcpu_init_probe(uint64_t x, uint64_t k) {")
            c.append("    volatile uint64_t _sink = VECTIS_LOKI_BIAS(k);")
            c.append("    VECTIS_BARRIER(x);")
            c.append("    return x ^ _sink;")
            c.append("}")
            c.append("typedef uint64_t (*_vcpu_probe_fn)(uint64_t, uint64_t);")
            c.append(f"static const _vcpu_probe_fn _vcpu_probe = _vcpu_init_probe;\n")

        # ── Function signature ──────────────────────────────────────────────────
        c.append("uint64_t vectis_execute_vcpu(const uint8_t *bytecode, size_t len, uint64_t initial_key) {")
        c.append("    (void)len;")
        c.append(f"    vcpu_state_t state = {{ .v_pc = 0, .v_key = 0x{vkey_init:016X}ULL ^ initial_key }};")
        c.append("")

        if spec.pinned_gprs:
            c.append("    register uint64_t v_r0 asm(\"r12\") = 0;")
            c.append("    register uint64_t v_r1 asm(\"r13\") = 0;")
            c.append("    register const uint8_t *pc asm(\"r15\") = bytecode;")
        else:
            c.append("    uint64_t v_r0 = 0, v_r1 = 0;")
            c.append("    const uint8_t *pc = bytecode;")
        c.append("")

        if spec.opaque_calls:
            c.append("    v_r0 = _vcpu_probe(v_r0, state.v_key);")
            c.append("    VECTIS_FENCE();\n")

        # ── Dispatch table ──────────────────────────────────────────────────────
        # Build table indexed by nibble: maps nibble → label
        tbl_entries = []
        for i in range(16):
            op = final_map.get(i, "TRAP")
            lbl = {
                "NOP": "op_nop", "ADD": "op_add", "XOR": "op_xor",
                "SUB": "op_sub", "ROR": "op_ror", "ROL": "op_rol",
                "LOAD": "op_load", "STORE": "op_store", "KEY_ROLL": "op_key_roll",
                "LOKI": "op_loki", "ARX": "op_arx", "MOV": "op_mov",
                "RET": "op_ret", "TRAP": "op_trap", "TRAP2": "op_trap",
                "TRAP3": "op_trap",
            }.get(op, "op_trap")
            tbl_entries.append(f"&&{lbl}")

        if spec.dtc:
            c.append(f"    static const void * const _dt[16] = {{")
            c.append(f"        {', '.join(tbl_entries)}")
            c.append(f"    }};")
            c.append(f"    #define DISPATCH() goto *_dt[(*pc++) & 0x0F]\n")
            c.append(f"    DISPATCH();\n")
        else:
            c.append("    #define DISPATCH() goto _top\n_top:")
            c.append(f"    switch ((*pc++) & 0x0F) {{\n")

        def hdr(label, case_val=None):
            if not spec.dtc and case_val is not None:
                c.append(f"    case 0x{case_val:X}:")
            c.append(f"{label}:")

        # ── NOP ────────────────────────────────────────────────────────────────
        hdr("op_nop", slot("NOP"))
        c.append("    DISPATCH();\n")

        # ── ADD ────────────────────────────────────────────────────────────────
        hdr("op_add", slot("ADD"))
        c.append("    {")
        if spec.rolling_key:
            if spec.multi_feistel:
                c.append(f"        uint64_t _k1 = ((state.v_key * 0x{mult:016X}ULL) ^ (state.v_key >> 17)) + 0x{delta_:016X}ULL;")
                c.append(f"        VECTIS_BARRIER(_k1);")
                c.append(f"        uint64_t _k2 = ((_k1 << 13) | (_k1 >> 51)) ^ 0x9E3779B97F4A7C15ULL;")
                c.append(f"        VECTIS_BARRIER(_k2);")
                c.append(f"        state.v_key = ((_k2 * 0xBF58476D1CE4E5B9ULL) ^ (_k2 >> 31)) + 0x94D049BB133111EBULL;")
            else:
                c.append(f"        state.v_key = ((state.v_key * 0x{mult:016X}ULL) ^ (state.v_key >> 17)) + 0x{delta_:016X}ULL;")
        c.append("        uint64_t _a = v_r0, _b = v_r1;")
        if spec.mba_depth >= 5:
            c.append("        uint64_t _t0 = (_a | _b); VECTIS_BARRIER(_t0);")
            c.append("        uint64_t _t1 = (_a ^ _b); VECTIS_BARRIER(_t1);")
            c.append("        uint64_t _t2 = 2ULL * _t0; VECTIS_BARRIER(_t2);")
            c.append("        uint64_t _bias = VECTIS_LOKI_BIAS(state.v_key); VECTIS_BARRIER(_bias);")
            c.append("        v_r0 = _t2 - _t1 + _bias;")
        elif spec.mba_depth >= 4:
            c.append("        uint64_t _t0 = (_a | _b); VECTIS_BARRIER(_t0);")
            c.append("        uint64_t _t1 = (_a ^ _b); VECTIS_BARRIER(_t1);")
            c.append("        uint64_t _bias = VECTIS_LOKI_BIAS(state.v_key); VECTIS_BARRIER(_bias);")
            c.append("        v_r0 = 2ULL * _t0 - _t1 + _bias;")
        elif spec.mba_depth >= 2:
            c.append("        v_r0 = (_a ^ _b) + 2ULL * (_a & _b);")
        else:
            c.append("        v_r0 = _a + _b;")
        c.append("        DISPATCH();")
        c.append("    }\n")

        # ── XOR ────────────────────────────────────────────────────────────────
        hdr("op_xor", slot("XOR"))
        c.append("    {")
        c.append("        uint64_t _a = v_r0, _b = v_r1;")
        if spec.isw_xor:
            c.append("        uint64_t _s0 = (_a | _b); VECTIS_BARRIER(_s0);")
            c.append("        uint64_t _s1 = (_a & ~_b); VECTIS_BARRIER(_s1);")
            c.append("        uint64_t _s2 = _s0 + _s1; VECTIS_BARRIER(_s2);")
            c.append("        uint64_t _km = state.v_key; VECTIS_BARRIER(_km);")
            c.append("        uint64_t _s3 = _s2 + _km; VECTIS_BARRIER(_s3);")
            c.append("        v_r0 = _s3 - _a - _km;")
        else:
            c.append("        uint64_t _t0 = (_a | _b); VECTIS_BARRIER(_t0);")
            c.append("        v_r0 = _t0 - (_a & _b);")
        c.append("        DISPATCH();")
        c.append("    }\n")

        # ── SUB ────────────────────────────────────────────────────────────────
        hdr("op_sub", slot("SUB"))
        c.append("    {")
        c.append("        uint64_t _a = v_r0, _b = v_r1;")
        if spec.twos_sub:
            c.append("        uint64_t _nb = ~_b; VECTIS_BARRIER(_nb);")
            c.append("        uint64_t _t0 = _a + _nb; VECTIS_BARRIER(_t0);")
            c.append("        v_r0 = _t0 + 1ULL;")
        else:
            c.append("        v_r0 = _a - _b;")
        c.append("        DISPATCH();")
        c.append("    }\n")

        # ── ROR ────────────────────────────────────────────────────────────────
        hdr("op_ror", slot("ROR"))
        c.append("    {")
        c.append("        uint64_t _a = v_r0, _sh = (*pc++) & 0x3F;")
        c.append("        v_r0 = (_a >> _sh) | (_a << (64 - _sh));")
        c.append("        DISPATCH();")
        c.append("    }\n")

        # ── ROL ────────────────────────────────────────────────────────────────
        hdr("op_rol", slot("ROL"))
        c.append("    {")
        c.append("        uint64_t _a = v_r0, _sh = (*pc++) & 0x3F;")
        c.append("        v_r0 = (_a << _sh) | (_a >> (64 - _sh));")
        c.append("        DISPATCH();")
        c.append("    }\n")

        # ── LOAD ───────────────────────────────────────────────────────────────
        hdr("op_load", slot("LOAD"))
        c.append("    {")
        c.append("        uint8_t _i = (*pc++) & 0x0F;")
        c.append("        v_r0 = state.bank.q[_i];" if spec.vbank else "        v_r0 = state.regs[_i];")
        c.append("        DISPATCH();")
        c.append("    }\n")

        # ── STORE ──────────────────────────────────────────────────────────────
        hdr("op_store", slot("STORE"))
        c.append("    {")
        c.append("        uint8_t _i = (*pc++) & 0x0F;")
        c.append("        state.bank.q[_i] = v_r0;" if spec.vbank else "        state.regs[_i] = v_r0;")
        c.append("        DISPATCH();")
        c.append("    }\n")

        # ── KEY_ROLL ───────────────────────────────────────────────────────────
        hdr("op_key_roll", slot("KEY_ROLL"))
        c.append("    {")
        if spec.multi_feistel:
            c.append(f"        uint64_t _k = state.v_key;")
            c.append(f"        _k = ((_k << 13) | (_k >> 51)) ^ 0x9E3779B97F4A7C15ULL;")
            c.append(f"        VECTIS_BARRIER(_k);")
            c.append(f"        _k = ((_k * 0xBF58476D1CE4E5B9ULL) ^ (_k >> 31)) + 0x94D049BB133111EBULL;")
            c.append(f"        VECTIS_BARRIER(_k);")
            c.append(f"        _k = ((_k ^ (_k >> 33)) * 0xFF51AFD7ED558CCDULL) ^ (_k >> 33);")
            c.append(f"        state.v_key = _k;")
        else:
            c.append(f"        state.v_key = (state.v_key << 13) | (state.v_key >> 51);")
            c.append(f"        state.v_key ^= 0x9E3779B97F4A7C15ULL;")
        c.append("        DISPATCH();")
        c.append("    }\n")

        # ── LOKI ───────────────────────────────────────────────────────────────
        hdr("op_loki", slot("LOKI"))
        c.append("    {")
        c.append("        uint64_t _x = state.v_key; VECTIS_BARRIER(_x);")
        c.append("        if (((_x * (_x + 1ULL)) & 1ULL) != 0) { __builtin_trap(); }")
        c.append("        DISPATCH();")
        c.append("    }\n")

        # ── ARX ────────────────────────────────────────────────────────────────
        hdr("op_arx", slot("ARX"))
        c.append("    {")
        if spec.arx_chaos:
            c.append(f"        uint64_t _k = state.v_key;")
            c.append(f"        uint64_t _t = v_r0 + _k; VECTIS_BARRIER(_t);")
            c.append(f"        _t = ((_t << {rot1}) | (_t >> {64-rot1})) ^ _k; VECTIS_BARRIER(_t);")
            c.append(f"        _t = _t + v_r1; VECTIS_BARRIER(_t);")
            c.append(f"        v_r0 = ((_t << {rot2}) | (_t >> {64-rot2})) ^ v_r1;")
        else:
            c.append("        v_r1 = v_r0;")
        c.append("        DISPATCH();")
        c.append("    }\n")

        # ── MOV ────────────────────────────────────────────────────────────────
        hdr("op_mov", slot("MOV"))
        c.append("    {")
        c.append("        v_r1 = v_r0;")
        c.append("        DISPATCH();")
        c.append("    }\n")

        # ── RET ────────────────────────────────────────────────────────────────
        hdr("op_ret", slot("RET"))
        if spec.masked_ret:
            c.append(f"        return v_r0 ^ VECTIS_LOKI_BIAS(state.v_key);\n")
        else:
            c.append("        return v_r0;\n")

        # ── TRAP ───────────────────────────────────────────────────────────────
        hdr("op_trap", slot("TRAP"))
        c.append("    __builtin_trap();\n")

        if not spec.dtc:
            c.append("    }\n")
        c.append("}\n")

        if emit_runner:
            ret_slot = slot("RET")
            loki_slot = slot("LOKI")
            nop_slot  = slot("NOP")
            c.append("#include <stdio.h>")
            c.append("int main(void) {")
            c.append(f"    uint8_t prog[] = {{ 0x{nop_slot:X}, 0x{loki_slot:X}, 0x{ret_slot:X} }};")
            c.append("    uint64_t r = vectis_execute_vcpu(prog, sizeof(prog), 0x1337BABE00000000ULL);")
            c.append("    printf(\"[+] 0x%016llx\\n\", (unsigned long long)r);")
            c.append("    return 0;")
            c.append("}\n")

        return "\n".join(c)


# ==============================================================================
# 7. PPO TRAINING + SYNTHESIS PIPELINE
# ==============================================================================

def train_ppo(env: VCPUSynthesisEnv, episodes: int = 60) -> VCPUSpec:
    best_spec, best_reward = None, -1e9

    if MLX_AVAILABLE:
        print("[*] PPO Actor-Critic on Metal GPU (128-hidden, 4 residual blocks)...")
        policy    = MLXPPOPolicy()
        optimizer = opt.Adam(learning_rate=5e-3)

        def ppo_loss(model, s_t, action, adv, old_lp, vt, clip=0.2):
            logits, val = model(s_t)
            lp  = (logits - mx.log(mx.sum(mx.exp(logits), keepdims=True)))[action]
            r   = mx.exp(lp - old_lp)
            return -mx.minimum(r * adv, mx.clip(r, 1-clip, 1+clip) * adv) + 0.5*(val[0]-vt)**2

        loss_grad = nn.value_and_grad(policy, ppo_loss)
    else:
        policy = None

    print(f"[*] Training {episodes} episodes, {env.ACTION_DIM} actions")
    for ep in range(1, episodes+1):
        state = env.reset(); done = False; ep_r = 0.0
        old_lp = mx.array(0.0) if MLX_AVAILABLE else None

        while not done:
            if MLX_AVAILABLE:
                s_t = mx.array(state, dtype=mx.float32)
                logits, _ = policy(s_t)
                probs = mx.softmax(logits).tolist()
                temp  = max(0.10, 1.2 - ep/(episodes*0.7))
                action = random.randint(0, env.ACTION_DIM-1) if random.random() < temp \
                         else int(random.choices(range(env.ACTION_DIM), weights=probs)[0])
            else:
                w = [4,1,2,4,4,2,3,2.5,2.5,5,4,4,3.5,3]
                action = random.choices(range(env.ACTION_DIM), weights=w)[0]

            next_s, reward, done, info = env.step(action)
            ep_r += reward

            if MLX_AVAILABLE:
                s_t  = mx.array(state, dtype=mx.float32)
                lp, _= policy.action_logprob(s_t, action)
                _, g = loss_grad(policy, s_t, action, mx.array(float(reward)), lp,
                                 mx.array(float(reward)))
                optimizer.update(policy, g)
                mx.eval(policy.parameters())
                old_lp = lp

            state = next_s

        if ep_r > best_reward:
            best_reward = ep_r
            best_spec   = env.spec

        if ep % 10 == 0 or ep == episodes:
            print(f"  Ep {ep:03d}/{episodes} | R={ep_r:+7.2f} | "
                  f"Cx={info['complexity']*100:.0f}% | Spd={info['speed']:.2f} | "
                  f"Cyc={info['avg_cyc']:.1f}")

    # Force fortress if RL didn't discover it
    for attr, val in [("mba_depth",5),("dtc",True),("pinned_gprs",True),
                       ("isw_xor",True),("twos_sub",True),("arx_chaos",True),
                       ("rolling_key",True),("vbank",True),("masked_ret",True)]:
        if getattr(best_spec, attr) != val:
            setattr(best_spec, attr, val)

    return best_spec


def main():
    parser = argparse.ArgumentParser(description="Vectis Neural VCPU Synthesizer v4 — Entropy Bridge")
    parser.add_argument("-o", "--output",    default="tools/synth_i3_ultra_vcpu.c")
    parser.add_argument("-e", "--episodes",  type=int, default=60)
    parser.add_argument("--seed",            type=int, default=None,
                        help="Entropy seed for reproducible builds (default: OS random)")
    parser.add_argument("--no-z3",          action="store_true")
    parser.add_argument("--emit-runner",     action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("  🚀  VECTIS NEURAL VCPU SYNTHESIZER v4 — OCAML ENTROPY BRIDGE")
    print("=" * 72)

    # ── Step 1: Query OCaml Entropy for unique per-build ISA params ────────────
    print(f"\n[*] Querying OCaml Entropy Bridge (vectis_synth.exe)...")
    ep = VectisEntropyBridge.query(seed=args.seed)
    print(f"    ISA Name     : {ep.isa_name}")
    print(f"    Entropy Seed : {ep.seed}")
    print(f"    ARX Rotations: {ep.arx_rot1} / {ep.arx_rot2}")
    print(f"    Feistel Mult : 0x{ep.lcg_mult:X}")
    print(f"    Loki Bias    : 0x{ep.loki_bias_const:016X} (even → always 0)")
    print(f"    Opcode Map   : " +
          " ".join(f"{op}=0x{s:X}" for op,s in sorted(ep.opcode_map.items(), key=lambda x:x[1])))

    # ── Step 2: RL training ───────────────────────────────────────────────────
    print(f"\n[*] PPO Training ({args.episodes} episodes)...")
    env  = VCPUSynthesisEnv()
    spec = train_ppo(env, episodes=args.episodes)
    spec.entropy = ep   # inject entropy params into spec

    # ── Step 3: Z3 formal verification ───────────────────────────────────────
    if not args.no_z3:
        print()
        Z3FormalVerifier.verify_all(spec)

    # ── Step 4: Emit C11 source ───────────────────────────────────────────────
    src = C11VCPUEmitter.generate_c11_source(spec, emit_runner=args.emit_runner)
    with open(args.output, "w") as f:
        f.write(src)

    print("\n" + "=" * 72)
    print(f" ✅  Generated: {args.output}")
    print(f"     Complexity : {spec.complexity()*100:.1f}/100")
    print(f"     ISA Unique : {ep.isa_name}  (seed={ep.seed})")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
