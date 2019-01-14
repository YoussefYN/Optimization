import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import math


class Ellipsoid:

    def __init__(self, a, b, c, p, xc):
        self.a = a
        self.b = b
        self.c = c
        self.p = p
        self.xc = xc
        self.w = c
        fig = plt.figure()
        # Draw constraints
        self.ax = fig.add_subplot(1, 1, 1)
        plt.axis([-2, 6, -2, 6])
        x = np.linspace(0, 5)
        y = -(a[3][0] / a[3][1]) * x + b[3] / a[3][1]
        plt.plot(x, y)
        y = -(a[4][0] / a[4][1]) * x + b[4] / a[4][1]
        plt.plot(x, y)
        plt.axvline(x=1)

    def draw_ellipse(self, p, xc):
        vals, vecs = np.linalg.eig(p)
        e = pat.Ellipse(xy=xc, width=vals[0], height=vals[1], angle=math.asin(vecs[1, 0]), fill=False)
        self.ax.add_patch(e)

    def solve(self):
        p = self.p
        xc = self.xc
        n = 2
        w = self.w
        for i in range(60):
            self.draw_ellipse(p, xc)
            idxs = np.where(np.greater(np.matmul(self.a, xc), self.b))[0]
            if len(idxs) > 0:
                w = self.a[idxs[0]]
            else:
                w = self.c
            v = np.matmul(p, w) / np.sqrt(np.matmul(w, np.matmul(p, w)))
            xc = xc - v / (n + 1)
            p = (n * n / (n * n - 1)) * (p - ((2 * np.matmul(np.transpose([v]), [v])) / (n + 1)))
        return xc

# Constrains => A*X <= B
a = np.array([[-1, 0], [0, -1], [1, 0], [0, 1], [1, 1]])
b = np.array([0, 0, 1, 1, 1.5])
c = np.array([-1, -2])
p = np.array([[4, 0], [0, 4]])
xc = np.array([0, 0])
ellipse = Ellipsoid(a, b, c, p, xc)
print(ellipse.solve())
plt.show()
