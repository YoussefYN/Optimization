import cvxpy as cvx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def show_ellipse(A, b, color, ax):
    sigma = np.linalg.inv(A.dot(A.T))
    mu = b
    vals, vecs = np.linalg.eigh(sigma)
    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))
    # Eigenvalues give length of ellipse along each eigenvector
    w, h = 2 * np.sqrt(np.power(vals, -1))
    ellipse = Ellipse(mu, w, h, theta, color=color)  # color="k")
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(0.6)
    ax.add_artist(ellipse)
    plt.show()


def get_mve(a, b):
    m, n = a.shape

    # Create and solve the model
    A = cvx.Variable((n, n), symmetric=True)
    d = cvx.Variable((n))
    obj = cvx.Maximize(cvx.log_det(A))
    constrs = [cvx.norm(A * a[i, :], 2) + a[i].T * d <= b[i] for i in range(m)]
    prob = cvx.Problem(obj, constrs)
    prob.solve(solver=cvx.SCS, verbose=False, eps=1e-6)
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var", A.value, d.value)
    return A.value, d.value


_, ax = plt.subplots()

#Constraints: AX <= B
a = np.array([[1, 1], [0, -1], [-1, 0], [0, 1], [1, 0]])
b = np.array([1.5, 0, 0, 1, 1])
x = np.array([[0, 0], [1, 0], [1, 0.5], [0.5, 1], [0, 1], [0, 0]])

B, d = get_mve(a, b)
plt.plot(x[:, 0], x[:, 1])
show_ellipse(B, d, 'b', ax)
plt.show()
