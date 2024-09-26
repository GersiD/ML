import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

class Log_Reg_Manual:
    def __init__(self, xdims, ydims) -> None:
        self.w = torch.randn(xdims, ydims, requires_grad=True)
        self.b = torch.randn(ydims, requires_grad=True)
        self.lr = 1e-3

    def forward(self, x):
        return torch.sigmoid(x @ self.w + self.b)

    def loss(self, y_pred, y):
        return nn.BCELoss()(y_pred, y)

    def fit(self, X, Y, epochs):
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            l = self.loss(y_pred, Y)
            if l == float('inf') or l == float('-inf') or torch.any(torch.isnan(l)):
                print(f"Loss is {l}, breaking")
                break
            losses.append(l.item())
            l.backward()
            with torch.no_grad():
                self.w -= self.lr * self.w.grad
                self.b -= self.lr * self.b.grad
                self.w.grad.zero_()
                self.b.grad.zero_()
            if epoch % 10 == 0:
                print(f'epoch {epoch}: loss = {l}')
        return losses


class Log_Reg_Torch(nn.Module):
    def __init__(self, xdims, ydims) -> None:
        super(Log_Reg_Torch, self).__init__()
        self.linear = nn.Linear(xdims, ydims)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-4)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def loss(self, y_pred, y):
        return nn.BCELoss()(y_pred, y)

    def fit(self, X, Y, epochs):
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            l = self.loss(y_pred, Y)
            if l == float('inf') or l == float('-inf') or torch.any(torch.isnan(l)):
                print("Loss is inf, breaking")
                break
            losses.append(l.item())
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                print(f'epoch {epoch}: loss = {l}')
        return losses

def fit_manual(X, Y, epochs):
    print("Fitting manual logistic regression model")
    model = Log_Reg_Manual(X.shape[1], Y.shape[1])
    losses = model.fit(X, Y, epochs)
    # plot losses
    # plt.plot(losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss MSE')
    # plt.title('Loss vs. Epoch for Manual Log Regression')
    # plt.savefig('./plots/loss_vs_epoch_manual.pdf')
    # plt.show()
    return losses

def fit_torch(X, Y, epochs):
    print("Fitting PyTorch logistic regression model")
    model = Log_Reg_Torch(X.shape[1], Y.shape[1])
    losses = model.fit(X, Y, epochs)
    # plot losses
    # plt.plot(losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss MSE')
    # plt.title('Loss vs. Epoch for PyTorch Log Regression')
    # plt.savefig('./plots/loss_vs_epoch_torch.pdf')
    # plt.show()
    return losses

def main():
    torch.manual_seed(8675309)
    # Example of training a logistic regression model to predict probability of high wine quality
    df = pd.read_csv('../data/wine_quality.csv')
    # print(df[["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"]].head())
    X = df[["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"]]
    y = df[["quality"]]
    y = (y > 6).astype(int)
    X = torch.tensor(X.values, dtype=torch.float32)
    Y = torch.tensor(y.values, dtype=torch.float32)
    man_losses = fit_manual(X, Y, 1000)
    torch_losses = fit_torch(X, Y, 1000)
    plt.plot(man_losses, label='Manual')
    plt.plot(torch_losses, label='PyTorch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss MSE')
    plt.title('Loss vs. Epoch for Manual vs. PyTorch Logsitic Regression')
    plt.legend()
    plt.savefig('./plots/loss_vs_epoch_manual_vs_torch.pdf')
    plt.show()

if __name__ == "__main__":
    main()
