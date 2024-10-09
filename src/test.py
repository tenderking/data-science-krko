from matplotlib import pyplot as plt
from src.plots import plot_accuracy_vs_regularization

if __name__ == "__main__":
    plot_accuracy_vs_regularization([1, 2, 3], [4, 5, 6], [7, 8, 9])
    plt.savefig("reports/figures/accuracy_vs_regularization.png")
