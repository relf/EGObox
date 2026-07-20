import os

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import egobox as egx
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Comment out the following line to display the plot in a window
matplotlib.use("Agg")


xt = np.array([[0.0, 1.0, 1.5, 2.0, 3.0, 4.0]]).T
yt = np.array([[0.0, 1.0, 1.7, 1.5, 0.9, 1.0]]).T

gpx = egx.GpMix().fit(xt, yt)

num = 100
x = np.linspace(0.0, 4.0, num).reshape((-1, 1))

y = gpx.predict(x)
# estimated variance
s2 = gpx.predict_var(x)

fig, axs = plt.subplots(1)

# add a plot with variance
axs.plot(xt, yt, "o")
axs.plot(x, y)
axs.fill_between(
    np.ravel(x),
    np.ravel(y - 3 * np.sqrt(s2)),
    np.ravel(y + 3 * np.sqrt(s2)),
    color="lightgrey",
)
axs.set_xlabel("x")
axs.set_ylabel("y")
axs.legend(
    ["Training data", "Prediction", "Confidence Interval 99%"],
    loc="lower right",
)

plt.show()

os.makedirs("examples_out", exist_ok=True)
fig.savefig("examples_out/kriging.png", dpi=150)
