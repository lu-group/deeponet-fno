import numpy as np


class Advection:
    def __init__(self, height, width, c, xm, xa, xb):
        """PBC"""
        self.h = height  # height
        self.w = width  # width
        self.xm = xm  # center location
        self.c = c  # velocity
        self.xa = xa  # Left domain point
        self.xb = xb  # right domain point

    def solve(self, X):
        x = X[:, 0:1]
        t = X[:, 1:2]
        n = np.shape(x)[0]
        u = np.zeros_like(x)
        h = self.h
        w = self.w
        c = self.c
        xm = self.xm
        xa = self.xa
        xb = self.xb
        for i in range(n):
            na = (xm - w / 2.0 + c * t[i, 0] - xa) // (xb - xa)
            nb = (xm + w / 2.0 + c * t[i, 0] - xa) // (xb - xa)
            xl = xa + (xm - w / 2.0 + c * t[i, 0] - xa) - na * (xb - xa)
            xr = xa + (xm + w / 2.0 + c * t[i, 0] - xa) - nb * (xb - xa)
            if xl < xr:
                if x[i, 0] >= xl and x[i, 0] <= xr:
                    u[i, 0] = h
                else:
                    u[i, 0] = 0.0
            else:
                if x[i, 0] >= xl or x[i, 0] <= xr:
                    u[i, 0] = h
                else:
                    u[i, 0] = 0.0
        # for i in range(n):
        #    if (x[i,0]-c*t[i,0]>=xm-w/2.0 and x[i,0]-c*t[i,0]<=xm+w/2.0):
        #        u[i,0] = h
        #    else:
        #        u[i,0] = 0.0
        return u


class Advection_v2:
    def __init__(self, x0, x1, c1, w, h1, c2, a, h2):
        self.x0 = x0
        self.x1 = x1
        self.c1 = c1
        self.w = w
        self.h1 = h1
        self.c2 = c2
        self.a = a
        self.h2 = h2

    def solve(self, X):
        x = X[:, 0:1] - X[:, 1:2]
        dx = x - self.x0
        n = dx // (self.x1 - self.x0)
        x -= n * (self.x1 - self.x0)

        xl = self.c1 - self.w / 2
        xr = self.c1 + self.w / 2
        u1 = self.h1 * np.heaviside(x - xl, 0.5) * np.heaviside(xr - x, 0.5)
        u2 = np.sqrt(np.maximum(self.h2 ** 2 - self.a ** 2 * (x - self.c2) ** 2, 0))
        return u1 + u2
