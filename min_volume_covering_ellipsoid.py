# Given N points in Rn. Select k of them such that the volume of ellipsoid
# covering these particular selected k points is minimal (compare to other
# ellipsoids covering different subsets of k points of N).

# This is an approximate solution
# based on the paper:http://www.cis.jhu.edu/~parky/CEP-Publications/PWPS-JCGS1997.pdf
import cvxpy as cvx
import matplotlib.pyplot as plt
import numpy as np
# Generate data
from matplotlib.patches import Ellipse


def get_optimal_ids(x, k):
    ids = np.array(range(len(x)))
    X_c = x

    while len(X_c) > k:
        X_c = X_c - X_c.mean(axis=0)
        eid = np.diag(np.matmul(X_c, np.matmul(np.linalg.inv(np.matmul(X_c.T, X_c)), X_c.T)))
        id = np.argmax(eid)
        ids = np.delete(ids, id)
        X_c = np.delete(X_c, id, axis=0)
    return ids


def show_ellipse(A, b, color, ax):
    sigma = np.linalg.inv(A.T.dot(A))
    mu = -sigma.dot(A.T).dot(b)
    vals, vecs = np.linalg.eigh(sigma)

    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Eigenvalues give length of ellipse along each eigenvector
    w, h = 2 * np.sqrt(vals)
    ellipse = Ellipse(mu, w, h, theta, color=color)  # color="k")
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(0.2)
    ax.add_artist(ellipse)
    plt.show()


def get_mve(x):
    n, m = x.shape

    # Create and solve the model
    A = cvx.Variable((n, n), symmetric=True)
    b = cvx.Variable((n))
    obj = cvx.Minimize(-cvx.log_det(A))
    constrs = [cvx.norm(A * x[:, i] + b, 2) <= 1 for i in range(m)]
    prob = cvx.Problem(obj, constrs)
    prob.solve(solver=cvx.SCS, verbose=False, eps=1e-6)
    return A.value, b.value, prob.value

X = np.random.randint(low=0, high=20, size=(20, 2))
ids = get_optimal_ids(X, 5)
print("Optimal points:\n", X[ids])
A, b, prob = get_mve(X[ids].T)
fig = plt.figure(0)
ax = fig.add_subplot(111, aspect='equal')
plt.scatter(X[:, 0], X[:, 1])
show_ellipse(A, b, 'g', ax)
