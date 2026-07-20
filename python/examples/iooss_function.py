import numpy as np
import matplotlib.pyplot as plt

import egobox as egx


def iooss_function(x):
    return (
        (np.exp(x[:, 0]) - x[:, 1]) / 5.0
        + x[:, 1] ** 6 / 3.0
        + 4.0 * (x[:, 1] ** 4 - x[:, 1] ** 2)
        + 7 * x[:, 0] ** 2 / 10.0
        + x[:, 0] ** 4
        + 3.0 / (4 * (x[:, 0] ** 2 + x[:, 1] ** 2) + 1.0)
    )


# Gp model
xt = egx.lhs([[-1.0, 1.0], [-1.0, 1.0]], 30)
yt = iooss_function(xt)

gpx = egx.Gpx.builder().fit(xt, yt)


# True function
x1 = np.linspace(-1, 1, 20)
x2 = np.linspace(-1, 1, 20)
X, Y = np.meshgrid(x1, x2)
xtrue = np.array(list(zip(X.reshape((-1,)), Y.reshape((-1,)))))
ytrue = iooss_function(xtrue)

# Prediction
ypred = gpx.predict(xtrue)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_trisurf(xtrue[:, 0], xtrue[:, 1], ytrue)
ax.scatter(xtrue[:, 0], xtrue[:, 1], ypred)

plt.title("2D Surface Heatmap")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.axis("equal")
plt.show()

# Save model
gpx.save("iooss_function.bin")
