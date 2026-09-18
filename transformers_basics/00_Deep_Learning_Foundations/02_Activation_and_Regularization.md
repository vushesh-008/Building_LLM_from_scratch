## 2. Activation Functions and Regularization

As a neural network gets deeper, it encounters two major problems:
1. **Mathematical instability during training** (Gradients becoming too large or too small).
2. **Overfitting** (The network simply memorizes the training data instead of learning general patterns).

To solve these, we must carefully choose our **Activation Functions** and employ **Regularization** techniques.

---

### Part 1: Activation Functions

An activation function introduces non-linearity into the network. 

**Why is non-linearity important?**
If you don't use an activation function, every layer in the network is just performing a linear transformation (multiplying by weights and adding biases). If you stack 100 linear layers on top of each other, the math mathematically simplifies down to a single linear layer. The deep network becomes no more powerful than a simple line of best fit, and it will be completely incapable of learning complex patterns like images or language.

By passing the output through a non-linear curve, the network can bend and twist its decision boundary to fit incredibly complex data.

#### 1. Sigmoid
The Sigmoid function squashes inputs into a smooth curve ranging between `0` and `1`. It was very popular in early AI because it mimics biological neurons (which are either "firing" or "not firing").
$$a = \sigma(z) = \frac{1}{1 + e^{-z}}$$
**Derivative:** $da = \sigma(z) \cdot (1 - \sigma(z))$

#### 2. Tanh (Hyperbolic Tangent)
Tanh is mathematically similar to Sigmoid but squashes inputs between `-1` and `1`. This is generally strictly superior to Sigmoid because it centres the data around zero, making the next layer's learning much easier.
$$a = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
**Derivative:** $da = 1 - (\tanh(z))^2$

#### The Problem: Vanishing Gradients
Why are Sigmoid and Tanh rarely used in deep networks today? Because of the **Vanishing Gradient Problem**. 

Look at the derivative of the Sigmoid function: $da = \sigma(z)(1 - \sigma(z))$. 
The absolute maximum possible value of this derivative is `0.25` (when the input is exactly 0). 

During backpropagation, the chain rule requires us to multiply the derivatives of each layer together. If you have a 100-layer network, you are multiplying numbers that are at most `0.25` together 100 times.
$$0.25 \times 0.25 \times 0.25 \dots = 0.0000000000001$$

By the time the error signal reaches the early layers of the network, the gradient is so microscopically small that the weights effectively don't update. The early layers freeze and fail to learn anything.

#### 3. ReLU (Rectified Linear Unit)
To fix this, modern networks almost exclusively use **ReLU** for their hidden layers. 
$$a = \max(0, z)$$
It is a simple hinge: If the input is negative, it outputs `0`. If the input is positive, it passes it through unchanged.

**Why is this a breakthrough?**
If $z > 0$, the derivative is exactly `1`.
When you chain rule `1` together 100 times, it stays `1`! The gradient does not vanish, allowing us to train incredibly deep networks.

**The Downside of ReLU:**
1. **Exploding Gradients:** Because gradients don't vanish, they can instead grow exponentially large in some cases. We solve this using **Gradient Clipping** (setting a hard speed limit on the maximum gradient value).
2. **Dead Neurons:** If a neuron's weights update such that it always outputs a negative number for every input in the dataset, its ReLU output becomes `0`, and its gradient becomes `0`. It can never recover and permanently "dies". 

*(Note: **Leaky ReLU** solves the dead neuron problem by allowing a tiny negative slope instead of a flat zero: $a = \max(0.01z, z)$)*

---

### Part 2: Regularization (Solving Overfitting)

When evaluating an AI model, we look at two metrics:
1. **Bias (Underfitting):** The model is too simple. It performs terribly on the training data. The solution is to train longer or build a bigger network.
2. **Variance (Overfitting):** The model performs flawlessly on the training data, but terribly on new data. It has literally memorized the training set. The solution is **Regularization**.

#### 1. L2 Regularization (Weight Decay)
L2 Regularization mathematically penalizes the network for relying too heavily on any single weight. It does this by adding the **Frobenius Norm** (the sum of the squared weights) to the cost function.

$$J_{reg} = J(w,b) + \frac{\lambda}{2m} \|W\|_F^2$$

During backpropagation, this naturally shrinks the weights on every single step:
$$w_{\text{new}} = w_{\text{old}} (1 - \frac{\alpha \lambda}{m}) - \alpha \cdot dw$$
Because we constantly multiply $w$ by a fraction slightly less than 1, this is also known as **Weight Decay**. It forces the network to keep all its weights small and distributed.

#### 2. Dropout Regularization
Dropout is a highly effective, seemingly insane form of regularization. 

During every pass of training, we set a probability (e.g., `keep_prob = 0.8`) and **randomly turn off 20% of the neurons** in the network. 

*Why on earth does destroying our own network help it learn?*
Imagine a team of 10 workers trying to build a house, but one worker is a genius who does everything while the other 9 slack off. If the genius gets sick, the house doesn't get built. 
Dropout randomly fires workers every day. The network can no longer rely on any single "genius" feature or neuron, because that neuron might be randomly dropped. It is forced to spread out its knowledge and learn redundant, robust features across all neurons.

#### 3. Other Methods
1. **Data Augmentation:** If your model is memorizing images of cats, flip the images horizontally, zoom in, or distort the colors. You have artificially doubled your dataset size, making it much harder to memorize.
2. **Early Stopping:** Track the error on a hidden "Dev Set" during training. As the model trains, the training error will drop forever. However, at some point, the Dev Set error will hit a minimum and begin to rise (this is the exact moment the model stops learning and starts memorizing). Stop the training right there.
