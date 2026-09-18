import egobox as egx
import numpy as np

print("Test: Multi-expert update")
x_train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
y_train = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0])

gpx = egx.Gpx.builder().fit(x_train, y_train)

print(f"Initial training data shape: {gpx.training_data()[0].shape}")
print(f"Initial theta values: {gpx.thetas()}")

# Update with new data
x_new = np.array([[5.5], [6.0]])
y_new = np.array([30.25, 36.0])

gpx_updated = gpx.update(x_new, y_new)
print(f"Updated training data shape: {gpx_updated.training_data()[0].shape}")
print(f"Updated theta values: {gpx_updated.thetas()}")

# Verify theta is unchanged (fixed theta approach)
assert np.allclose(gpx.thetas(), gpx_updated.thetas()), "Theta should remain unchanged!"
print("✓ Theta values remain unchanged (fixed theta)")

# Test prediction
x_test = np.array([[2.5], [4.5]])
y_pred = gpx_updated.predict(x_test)
y_expected = x_test.flatten() ** 2
print(f"Predictions: {y_pred.flatten()}")
print(f"Expected: {y_expected}")

# Check accuracy
error = np.abs(y_pred.flatten() - y_expected)
print(f"Absolute error: {error}")
assert np.all(error < 0.01), "Predictions should be accurate!"
print("✓ Predictions are accurate")

print("\n✓ All tests passed!")
