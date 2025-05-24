import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class NN_Torch(nn.Module):
    def __init__(self, xdims, ydims) -> None:
        super(NN_Torch, self).__init__()
        self.linear1 = nn.Linear(xdims, 64)
        self.linear2 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.2)
        self.linear3 = nn.Linear(64, ydims)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.dropout(x)
        return self.linear3(x)

    def loss(self, y_pred, y):
        return nn.functional.mse_loss(y_pred, y)

    def fit(self, X, Y, epochs):
        losses = []
        self.train()
        for epoch in range(epochs):
            y_pred = self.forward(X)
            l = self.loss(y_pred, Y)
            if l == float('inf') or l == float('-inf') or torch.any(torch.isnan(l)):
                print("Loss is inf, breaking")
                break
            losses.append(l.item())
            l.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if epoch % 10 == 0:
                print(f'epoch {epoch}: loss = {l}')
        self.eval()
        return losses

def fit_torch(X, Y, epochs):
    print("Fitting PyTorch NN model")
    model = NN_Torch(X.shape[1], Y.shape[1])
    losses = model.fit(X, Y, epochs)
    # plot losses
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss MSE')
    plt.title('Loss vs. Epoch for PyTorch NN')
    plt.savefig('./plots/loss_vs_epoch_torch.pdf')
    plt.show()
    return model, losses

def assess_torch(X, Y, model):
    print("Assessing PyTorch NN model")
    y_pred = model.forward(X)
    accuracy = nn.functional.mse_loss(y_pred, Y)
    print(f"Accuracy: {accuracy.item()}")


def main():
    torch.manual_seed(8675309)
    # Example of training a NN model to predict probability of high wine quality
    df = pd.read_csv('../data/wine_quality.csv')
    # print(df[["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"]].head())
    X = df[["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"]]
    y = df[["quality"]]
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=8675309)
    X_train = torch.tensor(X_train.values, dtype=torch.float32)# pyright: ignore[reportAttributeAccessIssue]
    X_test = torch.tensor(X_test.values, dtype=torch.float32)# pyright: ignore[reportAttributeAccessIssue]
    Y_train = torch.tensor(Y_train.values, dtype=torch.float32)# pyright: ignore[reportAttributeAccessIssue]
    Y_test = torch.tensor(Y_test.values, dtype=torch.float32)# pyright: ignore[reportAttributeAccessIssue]
    model, losses = fit_torch(X_train, Y_train, epochs=100)
    assess_torch(X_test, Y_test, model)

if __name__ == "__main__":
    main()
