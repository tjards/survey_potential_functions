#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 21:52:54 2024

This program implements various potential functions


@author: tjards
"""

import numpy as np
import matplotlib.pyplot as plt

# Define potential functions

# Lennard-Jones Potential
def lennard_jones(r, A=1, B=1):
    return A/r**12 - B/r**6

# Morse Potential
def morse(r, D=1, beta=1, r0=1):
    return D * (np.exp(-2*beta*(r - r0)) - 2*np.exp(-beta*(r - r0)))

# Spring-Like Potential
def spring_like(r, k=1, r0=3):
    return 0.5 * k * (r - r0)**2

# Soft Repulsion Potential
def soft_repulsion(r, A=1, p=1, epsilon=1, q=1):
    return A / (r**p + epsilon)**q

# Double-Well Potential
def double_well(r, k=1, r1=1, r2=5):
    return k * (r - r1)**2 * (r - r2)**2

# Exponential-Attractive and Polynomial-Repulsive Potential
def exp_poly_repulsive(r, B=1, m=1, C=1, alpha=1):
    return B/r**m - C * np.exp(-alpha * r)

# Riesz Potential
def riesz(r, s=1):
    return 1 / r**s

# Periodic Potential
def periodic(x, y, A=1, B=2, k=1):
    return A * np.sin(k*x)**2 + B * np.cos(k*y)**2

# Gaussian Potential
def gaussian(r, A=1, alpha=1, r0=1):
    return A * np.exp(-alpha * (r - r0)**2)

# Logarithmic Potential
def logarithmic(r, A=1):
    return -A * np.log(r)

# Coulomb Potential
def coulomb(r, A=1):
    return A / r

# Set up the range of r values for plotting
r_values = np.linspace(100, 0.1, 1000)  # Avoid r = 0 to prevent division by zero

# Plot each potential function
plt.figure(figsize=(10, 6))

# Plot Lennard-Jones potential
plt.plot(r_values, lennard_jones(r_values), label="Lennard-Jones", color="b")

# Plot Morse potential
plt.plot(r_values, morse(r_values), label="Morse", color="g")

# Plot Spring-Like potential
plt.plot(r_values, spring_like(r_values), label="Spring-Like", color="r")

# Plot Soft Repulsion potential
plt.plot(r_values, soft_repulsion(r_values), label="Soft Repulsion", color="c")

# Plot Double-Well potential
plt.plot(r_values, double_well(r_values), label="Double-Well", color="m")

# Plot Exponential-Attractive and Polynomial-Repulsive potential
plt.plot(r_values, exp_poly_repulsive(r_values), label="Exponential-Attractive", color="y")

# Plot Riesz potential
plt.plot(r_values, riesz(r_values), label="Riesz", color="orange")

# Plot Periodic potential
theta = 1
plt.plot(r_values, periodic(r_values*np.cos(theta), r_values*np.sin(theta)), label="Periodic", color="purple", linestyle = '--')

# Plot Gaussian potential
plt.plot(r_values, gaussian(r_values), label="Gaussian", color="brown")

# Plot Logarithmic potential
plt.plot(r_values, logarithmic(r_values), label="Logarithmic", color="pink")

# Plot Coulomb potential
plt.plot(r_values, coulomb(r_values), label="Coulomb", color="darkblue")

# Customize the plot
plt.xlabel('r (distance)')
plt.ylabel('V(r) (potential)')
plt.title('Potential Functions')
plt.axis([0, 10, -3, 10])
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
