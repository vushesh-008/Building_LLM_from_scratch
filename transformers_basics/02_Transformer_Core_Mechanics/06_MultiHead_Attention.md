## 5. Multi-Head Attention

Running a single attention head limits the model to one type of relationship per layer. Multi-head attention runs $h$ independent attention operations in parallel, each able to specialise.

```mermaid
flowchart TD
    IN["X  [N × d_model]"]
    subgraph HEAD1 ["Head 1  —  W_Q¹ W_K¹ W_V¹"]
        MM1a["Q₁K₁ᵀ / √dk"]
        N1["softmax → A₁"]
        MM1b["A₁ · V₁  →  [N × d_k]"]
    end
    subgraph HEADh ["Head h  —  W_Qʰ W_Kʰ W_Vʰ"]
        MMha["Qₕkₕᵀ / √dk"]
        Nh["softmax → Aₕ"]
        MMhb["Aₕ · Vₕ  →  [N × d_k]"]
    end
    CD["CONCAT  →  [N × d_model]\nthen  × W_O  [d_model × d_model]"]
    OUT["[y*₁, ..., y*_N]  [N × d_model]"]

    IN --> HEAD1 & HEADh
    HEAD1 --> CD
    HEADh --> CD
    CD --> OUT
```

**h → heads** (number of parallel attention operations)

Each head receives the full input $X$ but projects it through its **own separate** weight matrices $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$ — these are different for every head. Head 1 might learn to track coreference ("he" → "John"), head 2 might learn syntactic dependencies, etc.

The outputs of all heads are concatenated and passed through one final linear layer $W_O$ — this is the "Concat + Dense" step. Full geometry is in §6.

---
