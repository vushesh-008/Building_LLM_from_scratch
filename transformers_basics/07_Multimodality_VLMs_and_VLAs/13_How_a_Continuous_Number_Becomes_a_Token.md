## 11. How a Continuous Number Becomes a Token

Robot actions are continuous values (joint angles, velocities, gripper widths). The transformer vocabulary only contains discrete integers. Bridging this is **action tokenization via binning**.

### The binning process

Suppose one action dimension controls an arm's reach, range –1.0 m to +1.0 m.

**Step 1 — Define bins:**

```
  Range: [-1.0, +1.0]   →  divided into 256 equal bins

  Bin 0:    –1.000 to –0.992
  Bin 1:    –0.992 to –0.984
  ...
  Bin 128:  –0.008 to  0.000   ← roughly "centre / no movement"
  Bin 129:   0.000 to  0.008
  ...
  Bin 255:  +0.992 to +1.000
```

**Step 2 — Encode a real value:**

```
  Actual reach command:  +0.52 m

  Bin = floor( (0.52 - (-1.0)) / (2.0 / 256) )
       = floor( 1.52 / 0.0078125 )
       = floor( 194.56 )
       = bin 194

  Token ID assigned to bin 194:  vocab_id = 32000 + 194 = 32194
```

**Step 3 — At inference, decode back:**

```
  Model outputs token ID 32194
  → subtract text vocab offset: 32194 - 32000 = 194
  → bin 194 centre value: -1.0 + (194 + 0.5) × 0.0078125 = +0.516 m
  → send +0.516 m to arm controller
```

### A full action step — 7 dimensions

A typical robot arm has 6 DOF + gripper = 7 dimensions. Each gets its own bin token:

```
  Human demonstrator command at timestep t:
  [Δx=+0.52, Δy=–0.14, Δz=+0.03, Rroll=0.0, Rpitch=+0.21, Ryaw=–0.08, grip=CLOSE]

  Tokenised:
  [bin_194] [bin_110] [bin_132] [bin_128] [bin_155] [bin_118] [grip_close]
      ↑          ↑         ↑         ↑          ↑          ↑         ↑
    Δx         Δy        Δz       roll       pitch        yaw      gripper
```

The model predicts these 7 tokens one by one — standard next-token prediction, same as predicting the next word.

### Bin resolution tradeoff

```
  256 bins over 2m range  →  ~8mm precision per bin

  Coarser bins (fewer):   faster, less memory, but imprecise movements
  Finer bins (more):      more precise, but larger vocab, harder to learn

  Some papers use per-axis bin ranges:
    Gripper approach (needs precision): 512 bins over 0.1m → 0.2mm
    Gross transport (coarse OK):        128 bins over 2m   → 16mm
```

---
