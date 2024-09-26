import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

class Lin_Reg_Torch_Manual:
    def __init__(self, xdims, ydims) -> None:
        print(f'xdims = {xdims}, ydims = {ydims}')
        self.w = torch.randn(xdims, ydims, requires_grad=True)
        self.b = torch.randn(ydims, requires_grad=True)
        self.lr = 1e-5

    def forward(self, x):
        return x @ self.w + self.b

    def loss(self, y_pred, y):
        return ((y_pred - y) ** 2).mean()

    def backward(self, l):
        l.backward()
        with torch.no_grad():
            self.w -= self.lr * self.w.grad
            self.b -= self.lr * self.b.grad
            self.w.grad.zero_()
            self.b.grad.zero_()

    def fit(self, X, Y, epochs):
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            # print("y_pred.shape = ", y_pred.shape)
            l = self.loss(y_pred, Y)
            losses.append(l.item())
            self.backward(l)
            if epoch % 10 == 0:
                print(f'epoch {epoch}: loss = {l}')
        return losses

class Lin_Reg_Torch(nn.Module):
    def __init__(self, xdims, ydims) -> None:
        super(Lin_Reg_Torch, self).__init__()
        self.linear = nn.Linear(xdims, ydims)
        # pyright: ignore[reportPrivateImportUsage]
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-5)

    def forward(self, x):
        return self.linear(x)

    def loss(self, y_pred, y):
        return ((y_pred - y) ** 2).mean()

    def fit(self, X, Y, epochs):
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            l = self.loss(y_pred, Y)
            losses.append(l.item())
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                print(f'epoch {epoch}: loss = {l}')
        return losses


def fit_manual(X, Y, epochs):
    print("Fitting manual linear regression model")
    model = Lin_Reg_Torch_Manual(X.shape[1], Y.shape[1])
    losses = model.fit(X, Y, 50)
    # plot losses
    # plt.plot(losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss MSE')
    # plt.title('Loss vs. Epoch for Manual Linear Regression')
    # plt.savefig('./plots/loss_vs_epoch_manual.pdf')
    # plt.show()
    return losses

def fit_torch(X, Y, epochs):
    print("Fitting PyTorch linear regression model")
    model = Lin_Reg_Torch(X.shape[1], Y.shape[1])
    losses = model.fit(X, Y, 50)
    # plot losses
    # plt.plot(losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss MSE')
    # plt.title('Loss vs. Epoch for PyTorch Linear Regression')
    # plt.savefig('./plots/loss_vs_epoch_torch.pdf')
    # plt.show()
    return losses

def main():
    torch.manual_seed(8675309)
    assert os.getcwd().endswith('lin-reg-torch')

    # Example of training a linear regression model to predict wine quality
    # df = pd.read_csv('../data/wine_quality.csv')
    # # print(df[["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"]].head())
    # X = df[["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"]]
    # y = df[["quality"]]
    # X = torch.tensor(X.values, dtype=torch.float32)
    # Y = torch.tensor(y.values, dtype=torch.float32)

    # Example from https://www.kaggle.com/code/aakashns/pytorch-basics-linear-regression-from-scratch
    X = np.array([[73, 67, 43], 
                    [91, 88, 64], 
                    [87, 134, 58], 
                    [102, 43, 37], 
                    [69, 96, 70]], dtype='float32')
    Y = np.array([[56, 70], 
                        [81, 101], 
                        [119, 133], 
                        [22, 37], 
                        [103, 119]], dtype='float32')
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)
    manual_losses = fit_manual(X, Y, 50)
    torch_losses = fit_torch(X, Y, 50)
    # plot losses to compare
    plt.plot(manual_losses, label='Manual')
    plt.plot(torch_losses, label='PyTorch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss MSE')
    plt.title('Loss vs. Epoch for Manual vs. PyTorch Linear Regression')
    plt.legend()
    plt.savefig('./plots/loss_vs_epoch_manual_vs_torch.pdf')
    plt.show()

if __name__ == '__main__':
    main()
