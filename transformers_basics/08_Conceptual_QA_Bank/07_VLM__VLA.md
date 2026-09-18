## VLM / VLA

---

**Q: Why do image patch tokens get `label = –100` during VLA training if they still participate in attention?**

The two roles are independent. As **keys and values** in attention, image tokens are always active — every later token (including action tokens) can attend to them, read their content, and use that information to compute its output. That information flow is essential and happens regardless of labels.

As **prediction targets**, they would be asking the model to reconstruct the input image from scratch — a task the model was never designed to do and that adds no useful gradient signal for the robot control task. Setting their labels to –100 means `CrossEntropyLoss` ignores those positions, contributing zero to the gradient. They inform without being trained on.

```
  Image tokens as Keys/Values:  always active, carry visual info to all later positions  ✓
  Image tokens as Labels:       -100 → zero gradient, not a prediction target           ✗
```

---

**Q: Why does compounding error happen at inference but not during training?**

During training, every timestep in the demonstration trajectory is conditioned on the **ground-truth** previous state — the actual camera frame the human demonstrator was looking at, and the actual arm position at that moment. Even if the model would have made a wrong prediction at step 3, step 4 still receives the correct step-3 image as input. Errors never propagate.

At inference, the model conditions on **its own previous outputs**. A slightly wrong action bin at step 1 moves the arm to a slightly wrong position, which changes what the camera sees at step 2. The model now sees a camera frame it has likely never seen in training (it's off the demonstrated trajectory), and predicts the next action from an unfamiliar state. That prediction is slightly more wrong, moving the arm further off course — and so on.

```
  Training:   [GT image_t1] → action_t1 → [GT image_t2] → action_t2   (errors reset)
  Inference:  [image_t1]    → action_t1 → [wrong image_t2] → worse action_t2 → ...
```

This gap between training distribution and inference distribution is called **covariate shift**, and it's the fundamental challenge of behavioural cloning.

---

**Q: Why use discrete action bins rather than directly regressing continuous joint values?**

The standard transformer output is a probability distribution over a discrete vocabulary — that is the mechanism it was designed for and pre-trained on. Directly regressing a continuous value would require replacing the softmax head with a regression head (e.g. predicting a float with MSE loss), which:

1. Loses the pre-trained output head and requires training a new one from scratch
2. Changes the loss function (MSE vs cross-entropy) in ways that interact poorly with the rest of the fine-tuning
3. Discards the model's uncertainty representation — a probability distribution over bins naturally expresses "I'm not sure if the arm should go left or right" in a way a single float cannot

By binning, the VLA reuses the *exact same* prediction mechanism as language — a softmax over a vocabulary. The only change is that some vocabulary entries now mean "move 8mm in the x-direction" instead of "the word 'cat'." This lets the model transfer all of its pre-trained next-token prediction machinery directly to action prediction.

---

**Q: Why does VLA training need significantly more data than a similarly-sized text fine-tune?**

Two reasons compound each other:

**1. Supervision sparsity.** Only ~3% of tokens in a VLA training example get a gradient — the action bins. The image patches (often 196 tokens per frame) and state/instruction tokens are context only. A 10-step trajectory might have ~2170 tokens total but only 70 supervised tokens. The model is doing enormous amounts of "reading" for a very sparse reward signal.

**2. Physical diversity.** Text fine-tuning can generalise from limited examples because language has regular structure — patterns seen in one context transfer to others. Physical manipulation does not generalise as readily — a policy trained on mugs in one lighting condition, on one table surface, at one camera angle, may fail completely under minor perturbations. Every new object, background, or lighting condition is effectively a new distribution that needs to be covered by training data.

Combined: the model needs to see enough diverse trajectories that both the action token supervision accumulates meaningfully AND the image-state distribution is covered broadly enough to avoid the compounding error problem at inference.

---
