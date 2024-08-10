import matplotlib.pyplot as plt
import numpy as np
# Generate the circular dataset (same as before)
def generate_circles(n_samples=200, noise=0.2):
    np.random.seed(0)
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    theta = np.linspace(0, 2 * np.pi, n_samples_out)
    outer_circ_x = np.cos(theta)
    outer_circ_y = np.sin(theta)

    theta = np.linspace(0, 2 * np.pi, n_samples_in)
    inner_circ_x = 0.5 * np.cos(theta)
    inner_circ_y = 0.5 * np.sin(theta)

    X = np.vstack([
        np.append(outer_circ_x, inner_circ_x),
        np.append(outer_circ_y, inner_circ_y)
    ]).T
    y = np.hstack([np.zeros(n_samples_out), np.ones(n_samples_in)])

    X += noise * np.random.randn(*X.shape)
    y = y.reshape(-1, 1)

    return X, y

# Generate and plot the dataset
X, y = generate_circles(n_samples=200, noise=0.1)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=plt.cm.Spectral, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Circular Dataset Visualization')
plt.show()