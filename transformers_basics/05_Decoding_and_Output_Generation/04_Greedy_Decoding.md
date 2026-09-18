## 3. Greedy Decoding

At each step, pick the **single most probable token** and move on.

```
  Step 1:
  P = [0.35, 0.22, 0.18, 0.10, ...]
        "the"  "a"  "an" "one"
                │
                ▼  argmax
           → Pick "the"   ✓

  Step 2 (conditioned on "the"):
  P = [0.40, 0.25, 0.15, ...]
       "cat"  "dog" "big"
                │
                ▼  argmax
           → Pick "cat"   ✓

  Output: "the cat ..."
```

**Pros:** Fast, simple, deterministic.

**Cons:** Locally optimal ≠ globally optimal. Can get stuck in repetitive loops.

```mermaid
flowchart TD
    START["I"]

    START -->|"P=0.40 ✓ greedy picks"| LOVE["love"]
    START -->|"P=0.25 ✗ greedy ignores"| ADORE["adore"]

    LOVE -->|"most likely next"| YOU["you\n→ 'I love you'\n(generic)"]
    ADORE -->|"most likely next"| THE["the way you\nthink about things\n→ richer sentence"]

    style LOVE fill:#1a4a2e,color:#7ddfaa,stroke:#2e8a55
    style YOU fill:#1a4a2e,color:#7ddfaa,stroke:#2e8a55
    style ADORE fill:#5c1e2e,color:#ffb3c1,stroke:#c44d6b
    style THE fill:#5c1e2e,color:#ffb3c1,stroke:#c44d6b
```

> Greedy picks the locally best token at every step. The green path wins step 1, but the red path (ignored) leads to a globally better sequence.


---
