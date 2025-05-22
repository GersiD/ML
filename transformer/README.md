# Tansformer Model

Introduced in the [landmark paper from Google Research](https://arxiv.org/abs/1706.03762), the Transformer model is a neural network architecture that relies entirely on self-attention mechanisms to draw global dependencies between input and output. 
Unlike RNNs, which process data sequentially, Transformers can process all tokens in the input sequence simultaneously, making them highly parallelizable and efficient for training on large datasets.

Lately the Transformer architecture has been widely adopted in various natural language processing (NLP) tasks, including machine translation, text summarization, and sentiment analysis. This folder is dedicated to the implementation of the Transformer model, which is a key component of many state-of-the-art NLP systems.

Due to the complexity of the Transformer architecture the original work from Google Research is quite long and difficult to follow.
There has been much followup work on understanding the technical reasons for [why the Transformer architecture works so well](https://nlp.seas.harvard.edu/2018/04/03/attention.html).

My implementation here follows the original paper closely, but I have also added some additional features and improvements based on my own research and experimentation. For a detailed guide on implementing the Transformer architecture in PyTorch, please see [harvard nlp lab](https://nlp.seas.harvard.edu/2018/04/03/attention.html).

## Task
In this implementation I train the transformer model on a simple task of given a random list of numbers returning the
same list of numbers in the same order. This is a simple task but it is a good way to test the implementation of the transformer model learning what is refered to as a identity transformation.

The main goal is to understand the conditions under which the transformer is likely to converge given the choice of hyperparameters and the size of the dataset. Further I vary the multi-attention heads and the number of layers in the encoder and decoder to see how this affects the performance of the model.

## Running the Code
To run the code, you will need to have Python 3.6 or later installed on your machine. You can install the required packages using pip:
```bash
python -m venv venv
source venv/bin/activate
pip install torch matplotlib numpy
```
Then, you can run the code by executing the following command in your terminal:
```bash
python transformer.py
```
Observing the following output:
```text
Epoch Step: 1 Loss: 2.447901 Tokens per Sec: 1422.573975
Epoch Step: 1 Loss: 1.532197 Tokens per Sec: 5133.891602
Epoch Step: 1 Loss: 1.353720 Tokens per Sec: 5087.316406
Epoch Step: 1 Loss: 1.191844 Tokens per Sec: 5139.367188
Epoch Step: 1 Loss: 0.833867 Tokens per Sec: 5155.921387
Epoch Step: 1 Loss: 0.573462 Tokens per Sec: 5161.749512
Epoch Step: 1 Loss: 0.409838 Tokens per Sec: 5177.206055
Epoch Step: 1 Loss: 0.324658 Tokens per Sec: 5147.402832
Epoch Step: 1 Loss: 0.374183 Tokens per Sec: 5127.325195
Epoch Step: 1 Loss: 0.217464 Tokens per Sec: 5176.815918
```

## Findings
The transformer model is able to learn the identity transformation quite quickly. The loss converges to a low value in a few epochs which is expected. Varying the number of multi-attention heads and the number of layers in the encoder and decoder significantly affects the convergance rate of the model which indicates that the choice of model complexity is important for the task at hand.

The trained model can be decomposed into a decoder only model which is able to aid in tasks such as text similarity, and
text classification. This can be accomplished using cosine distance on the vector embedding of the input sequence.
