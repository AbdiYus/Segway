import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import control as ctl

class KalmanFilter:

    def __init__(self, F, G, H, Q, R, x0, P0):
        self.F = F
        self.G = G
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self, u):
        self.x = self.F @ self.x + self.G @ u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, y, H=None, R=None):
        if H is None:
            H = self.H
        if R is None:
            R = self.R

        error = y - H @ self.x      

        error_size = H @ self.P @ H.T + R

        gain_K = self.P @ H.T @ np.linalg.inv(error_size)

        # Correction:
        self.x = self.x + gain_K @ error

        I = np.eye(len(self.x))

        # Update uncertainty
        self.P = (I - gain_K @ H) @ self.P





