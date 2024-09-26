# Method

Logistic regression is a common method used for binary classification problems.
It is a simple model that assumes a linear relationship between the input variables and the output variable in
exponential space. For more information see [GLM](https://en.wikipedia.org/wiki/Generalized_linear_model) and [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression).
The model is defined by the equation:

y = 1 / (1 + exp(-(b0 + b1*x1 + b2*x2 + ... + bn*xn)))

where y is the output variable, b0 is the bias term, and b1 to bn are the coefficients for the input variables x1 to xn.

# Running the code
Ensure the requirements are installed (see requirements.txt in the root project directory).

```bash
cd log-reg-torch/ # Change to the directory containing the code from the root project directory
python log-reg-torch.py
```

# Results
See the *./plots/loss_vs_epoch_manual_vs_torch.pdf* file for a comparison of the loss vs epoch for the manual and torch implementations of linear regression.

By manual I mean that the model is implemented from scratch using numpy and torch for gradient computation. 
And by torch I mean that the model is implemented using the torch.nn.Linear module.

```text
*Notice the poor performance of the manual implementation of logistic regression*. 
This is due to the fact that the gradient is small and the step size is not chosen well.
The torch implementation uses the SGD (stochastic gradient decent) optimizer which adapts the step size based on the gradient.
```
