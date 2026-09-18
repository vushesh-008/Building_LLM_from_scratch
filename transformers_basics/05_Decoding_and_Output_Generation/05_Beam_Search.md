## 4. Beam Search

Instead of keeping 1 sequence, keep the **top-B sequences** (beams) at every step.

```mermaid
flowchart TD
    S["Start"]

    S -->|"P=0.35 ✓ keep"| T["'the'\n0.350"]
    S -->|"P=0.22 ✓ keep"| A["'a'\n0.220"]
    S -->|"P=0.18 ✗ drop"| AN["'an'\n0.180"]

    T -->|"×0.40"| TC["'the cat'\n0.140 ✓ keep"]
    T -->|"×0.25"| TD["'the dog'\n0.088 ✓ keep"]
    T -->|"×0.15"| TB["'the big'\n0.053 ✗ drop"]

    A -->|"×0.38"| AC["'a cat'\n0.084 ✗ drop"]
    A -->|"×0.20"| AD["'a dog'\n0.044 ✗ drop"]

    TC --> FINAL["Final: pick highest scoring\ncomplete sequence\n→ 'the cat ...'"]
    TD --> FINAL

    style AN fill:#3a3a3a,color:#777,stroke:#555
    style TB fill:#3a3a3a,color:#777,stroke:#555
    style AC fill:#3a3a3a,color:#777,stroke:#555
    style AD fill:#3a3a3a,color:#777,stroke:#555
    style TC fill:#1a2e5c,color:#a8c4ff,stroke:#3d68c8
    style TD fill:#1a2e5c,color:#a8c4ff,stroke:#3d68c8
    style FINAL fill:#1a4a2e,color:#7ddfaa,stroke:#2e8a55
```

At each step, **all surviving beams expand** into their top candidates, then only the global top-B survive across all expansions.

![Beam search tree — paths kept vs dropped at each step](../images/beam_search_tree.svg)

**Pros:** Much better than greedy — explores multiple paths.

**Cons:** Still deterministic. Can produce generic/safe outputs. Expensive (B × compute per step).

---
