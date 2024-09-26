# Method

The method that changed the world of machine learning forever, [neural nets](https://en.wikipedia.org/wiki/Neural_network).
They are NP-hard to train and require a lot of data and computational power. 
The model is defined by a number of layers, each layer containing a number of neurons.
Each neuron is defined by the equation:

y = activation_function(b0 + b1*x1 + b2*x2 + ... + bn*xn)

where y is the output variable, b0 is the bias term, and b1 to bn are the coefficients for the input variables x1 to xn.

In a way NNs are a generalization of logistic regression and [GLMs](https://en.wikipedia.org/wiki/Generalized_linear_model), 
where we abstract from optimizing a fit in exponential space to some space defined by the activation_function.

# Running the code
Ensure the requirements are installed (see requirements.txt in the root project directory).

```bash
cd nn-torch/ # Change to the directory containing the code from the root project directory
python nn-torch.py
```

# Results
<!--TODO: Possible extension manual implementation of a NN-->
There is no manual implementation of neural nets, since implementing one is in and of itself a complicated endevour left
outside the scope of this work. However maybe sometime in the future this would be appropriate.

```info
It is encouraged that the reader download and play with the lr parameter in the code to see what the different learning
rates do to the loss curve :).
```
