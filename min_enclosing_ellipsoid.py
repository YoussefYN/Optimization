import cvxpy as cvx
import numpy as np
from descartes import PolygonPatch
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from shapely.geometry.polygon import Polygon


def get_ellipse(x):
    x = x.T
    n, m = x.shape
    # Create and solve the model
    A = cvx.Variable((n, n), symmetric=True)
    b = cvx.Variable((n))
    obj = cvx.Maximize(cvx.log_det(A))
    constrs = [cvx.norm(A * x[:, i] + b, 2) <= 1 for i in range(m)]
    prob = cvx.Problem(obj, constrs)
    prob.solve(solver=cvx.SCS, verbose=False, eps=1e-6)
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var", A.value, b.value)
    return A.value, b.value


def show_ellipse(A, b, color, ax):
    sigma = np.linalg.inv(A.dot(A.T))
    mu = np.linalg.inv(A).dot(-b)
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


# Generate data
xs = np.array([[0, 0], [1, 0], [1, .5], [.5, 1], [0, 1]])
# get Minimum enclosing volume ellipsoid
A, b = get_ellipse(xs)

fig = plt.figure(0)
ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])

show_ellipse(A, b, 'r', ax)
ring_mixed = Polygon(xs)
ring_patch = PolygonPatch(ring_mixed)
ax.add_patch(ring_patch)
plt.show()
