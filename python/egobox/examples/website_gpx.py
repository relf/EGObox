import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import egobox as egx


def fit_surrogate_example():
    xtrain = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ytrain = np.array([0.0, 1.0, 1.5, 0.9, 1.0])
    gpx = egx.Gpx.builder().fit(xtrain, ytrain)

    xtest = np.linspace(0.0, 4.0, 100).reshape((-1, 1))
    ytest = gpx.predict(xtest)

    fig, ax = plt.subplots()
    ax.plot(xtest, ytest)
    ax.plot(xtrain, ytrain, "o")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return gpx, xtest, ytest, fig


def main() -> None:
    _, _, _, fig = fit_surrogate_example()
    fig.savefig("website_gpx.png", dpi=150)


if __name__ == "__main__":
    main()
