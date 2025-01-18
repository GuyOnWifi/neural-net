Multi-layer perception from scratch; no Pytorch or Tensorflow. 96% accuracy on MNIST digits. 

# Motivation

Neural networks are a fantastic algorithm: they have so many free parameters that can be tuned to continously learn and improve. I wanted to get a strong, intuitive understanding of how these networks learn; what better way then coding one from scratch? No external ML libraries like PyTorch or TensorFlow, just numpy and a whole lot of linear algebra.

# Running the code
To run the website:
```bash
cd website
npm run dev
```
All of the neural network models are stored in `/src/`, with example uses stored in the `/examples/` folder. Try out `draw.py` to test the model on your handwritten digits!

# How it works

## Feedforward
The multilayer perceptron is made up of layers of neurons. Each neuron has an activation, which is essentially the output of the neuron. Each connection between neurons of two adjacent layers have a weight, and every output neuron has an associated bias. The weighted sums each neuron is the activation * weights of each connected node, with another biases added on. The weighted sum is put through an activation function to get the final activation result.

### Mathematical Representation of the Feedforward

Each activation is represented as a (n x 1) vector

$$
\mathbf{a} = \begin{pmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{pmatrix}
$$

Each weight is represented by a (m x n) matrix, where m is the number of output neurons and n is the number of input neurons.
Each weight $w_{i,j}$ represents the connection between the $j$-th input neuron to the $i$-th ouput neuron.

$$
\mathbf{w} = \begin{pmatrix} 

w_{1,1} & w_{1,2} & \cdots & w_{1,n} \\
w_{2,1} & w_{2,2} & \cdots & w_{2,n} \\
\vdots & \vdots & \ddots & \vdots \\
w_{m,1} & w_{m,2} & \cdots & w_{m,n}

\end{pmatrix}
$$

The bias can be represented as a (m x 1) matrix

$$
\mathbf{b} = \begin{pmatrix} b_1 \\ b_2 \\ \vdots \\ b_m \end{pmatrix}
$$

The weighted sums $z$ can be represented as the dot product of the activations and weights matrix, plus the bias matrix

$$
z = 
\begin{pmatrix} 

w_{1,1} & w_{1,2} & \cdots & w_{1,n} \\
w_{2,1} & w_{2,2} & \cdots & w_{2,n} \\
\vdots & \vdots & \ddots & \vdots \\
w_{m,1} & w_{m,2} & \cdots & w_{m,n}

\end{pmatrix}

\begin{pmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{pmatrix}
+
\begin{pmatrix} b_1 \\ b_2 \\ \vdots \\ b_m \end{pmatrix}
$$

which expands out to

$$
z =
\begin{pmatrix}
w_{1, 1} \cdot a_1 + w_{1, 2} \cdot a_2 + \cdots + w_{1, n} \cdot a_n + b_1 \\
w_{2, 1} \cdot a_1 + w_{2, 2} \cdot a_2 + \cdots + w_{2, n} \cdot a_n + b_2 \\
\vdots \\
w_{m, 1} \cdot a_1 + w_{m, 2} \cdot a_2 + \cdots + w_{m, n} \cdot a_n + b_m

\end{pmatrix}
$$

The output weighted sums is then put through something known as an **activation function**, to get the final activations. My model suports three: ReLU, sigmoid and softmax.

$$
ReLU(z) = max(0, z)
$$

$$
sigmoid(z) = \frac{1}{1+e^{-z}}
$$

$$
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$
The activations are used as the inputs to the next layer.

### Cost function
We can't talk about neural networks without some kind of **cost function**. A cost function computes the "cost" or the error from the expected values based on the final output activations. My model supports two cost functions: MSE (mean-squared-error) and cross-entropy.


Note that $y_i$ represents the actual, expected value for the $i$-th data point, and that $\hat{y_i}$ represents the value predicted by the model.

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

$$
\text{Cross-Entropy} = - \sum_{i=1}^{n} y_i \log(\hat{y}_i)
$$
The goal of any neural network is to minimize the cost.

## Backpropogation
The key to neural networks learning is the backpropogation algorithm. What it does is computes the gradient of the cost function $\nabla C$, which tells you how much the weights and biases need to change to decrease the cost function.
If that confuses you, think abut it this way: the gradient is just the slope of the tangent line to the equation. Say you had a curve and wanted to find a realtive minima. You can do that by calculating its derivitave (gradient in this case) and adjusting your x values until you reach the minimum.

The backpropogation algorithm, as you can probably already guess, requires differentiaton and calculus. Specifically, it uses a lot of the chain rule to determine gradients. 

One other important concept is the **learn rate**. The learn rate is the factor which we multiply our gradients by adding them onto our weights and biases. Remember, the derivative of a function only gives the instantaneous rate of change, so we'll need a learn rate that is small enough to give predictable behavior, but large enough so that training isn't too slow.

### Calculus symbols 
To start backpropogating, we'll first need the partial derivaitve of the last layer's activations to the cost function.

$$
\frac{\partial C}{\partial a}
$$

For each layer, we'll look at the partial derivative of the activations to the weights and biases by using the chain rule

$$
\frac{\partial a}{\partial w} = \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w_j}
$$

$$
\frac{\partial a}{\partial b} = \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial b}
$$

And to continue back propogation through layers, it's important to calculate the derivative of the previous layer's activation with respect to the current layer.

$$
\frac{\partial a}{\partial a_{\text{prev}}} = \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial a_{\text{prev}}}
$$

When we backpropogate, what we're essentially doing is
1. Find the sensitivity of the last layer's activations to the cost function
2. Repeat for all layers:
    1. Use the activation sensitivity, and compute the weight and bias gradients
    2. Find the sensitvity to last layer's activation
    3. Pass in that value as the next layers activation sensitivity
3. Update the weights and biases based on their gradients, multiplied by learn rate

# Further Improvements
- Convolutional layers
- More datasets like Google Draw
- Improving the README to include the actual math behind the calculus and L2 regularization, as well as vectorization of feedforward and backprop computations

# Resources
These are some great explanations for neural networks
- [3Blue1Brown's Playlist on Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) - amazing visual explanations
- [Michael Nielsen's Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - dives heavily into the math, shares python code (I'd recommend optimizing/vectorizing the code yourself - it does a great job explaining but doesn't take full advantage of linear algebra libraries)