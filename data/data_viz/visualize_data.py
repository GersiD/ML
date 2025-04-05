import os
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def plot_iris_data(iris_df):
    # cols : sepal length, sepal width, petal length, petal width, class
    # Plot each feature against each other in the same plot like R
    colors = iris_df['class'].map({'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3})
    plt.scatter([], [], label='Sepal', marker='o', color='black')
    plt.scatter(iris_df['sepal length'], iris_df['sepal width'], c=colors, marker='o')
    plt.scatter([], [], label='Petal', marker='x', color='black')
    plt.scatter(iris_df['petal length'], iris_df['petal width'], c=colors, marker='x')
    plt.title('Sepal and Petal Length vs Width')
    plt.xlabel('Length')
    plt.ylabel('Width')
    plt.legend()
    plt.colorbar(label='Class')
    plt.savefig('iris_plot.pdf')
    plt.clf()  # Clear the current figure for the next plot

def plot_wine_data(wine_df):
    # cols : ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
    # Plot each feature against each other, pair-plot
    g = sns.pairplot(wine_df, corner=True)
    g.figure.suptitle('Wine Quality Analysis', y=1.02)
    g.savefig('wine_plot.pdf')

def main():
    # Load the dataset
    iris_file_path = "../iris.csv"
    wine_file_path = "../wine_quality.csv"
    iris_df = pd.read_csv(iris_file_path)
    wine_df = pd.read_csv(wine_file_path)
    # Plot the iris dataset
    print("Iris dataset loaded")
    print("Iris dataset shape:", iris_df.shape)
    plot_iris_data(iris_df)
    # Plot the wine dataset
    print("Wine dataset loaded")
    print("Wine dataset shape:", wine_df.shape)
    plot_wine_data(wine_df)


if __name__ == "__main__":
    main()
