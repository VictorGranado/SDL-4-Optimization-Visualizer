# Optimization Visualizer — SDL 4 (Multivariable Calculus)

This project is an interactive Python application created for a Self-Directed Learning (SDL) project focused on **optimization in multivariable calculus**. It was designed to make abstract topics like gradients, directional derivatives, critical points, and constrained optimization easier to understand through visual and symbolic exploration.

Instead of only solving problems by hand, this tool allows users to graph functions, inspect derivative information, and see how optimization ideas behave geometrically. The goal is to connect equations to intuition by showing surfaces, contour plots, vectors, and symbolic derivative information in one place.

---

## Project Purpose

This project was built to strengthen understanding of the main ideas from this unit, including:

- contour plots and 3D surfaces
- partial derivatives and gradients
- directional derivatives
- critical points and classification
- constrained optimization using Lagrange multipliers
- symbolic differentials, Hessians, and eigenvalues

The application is meant to function as both a study tool and a way to connect multivariable calculus concepts to visual interpretation.

---

## Application Overview

The program is organized into six main tabs.

---

### 1. Surface + Contours

This tab plots a function \(f(x,y)\) in two ways:

- as a **contour plot**
- as a **3D surface**

This helps visualize the overall shape of the function and makes it easier to identify patterns such as bowls, ridges, and saddles.

![Alt text](https://github.com/VictorGranado/SDL-4-Optimization-Visualizer/blob/400de97f20c49ae5d9386cd8fb7fe9135b2e52a7/Screenshot%202026-03-12%20181101.png)

---

### 2. Gradient Explorer

This section shows the gradient field of a function and highlights the gradient vector at a chosen point.

It helps explain:

- how partial derivatives combine into the gradient
- the direction of steepest increase
- how gradients relate to contour lines

![Alt text](https://github.com/VictorGranado/SDL-4-Optimization-Visualizer/blob/400de97f20c49ae5d9386cd8fb7fe9135b2e52a7/Screenshot%202026-03-12%20181115.png)

---

### 3. Directional Derivative

This tab allows the user to choose a point and a direction vector, then computes the directional derivative.

It displays:

- the gradient vector
- the chosen direction vector
- the numerical directional derivative

This makes it easier to understand how the rate of change depends on direction.

![Alt text](https://github.com/VictorGranado/SDL-4-Optimization-Visualizer/blob/400de97f20c49ae5d9386cd8fb7fe9135b2e52a7/Screenshot%202026-03-12%20181122.png)

---

### 4. Optimization (Critical Points)

This section finds candidate critical.

It then classifies them using the Hessian matrix and its eigenvalues.

Possible classifications include:

- local minimum
- local maximum
- saddle point

This tab helps connect derivative conditions to the geometry of the function.

![Alt text](https://github.com/VictorGranado/SDL-4-Optimization-Visualizer/blob/400de97f20c49ae5d9386cd8fb7fe9135b2e52a7/Screenshot%202026-03-12%20181131.png)

---

### 5. Lagrange Multipliers

This tab explores constrained optimization problems of the form:


text{optimize } f(x,y)  g(x,y)=c


It plots:

- contour lines of the objective function
- the constraint curve
- the candidate optimization points

This provides a visual interpretation of why Lagrange multipliers work.

![Alt text](https://github.com/VictorGranado/SDL-4-Optimization-Visualizer/blob/400de97f20c49ae5d9386cd8fb7fe9135b2e52a7/Screenshot%202026-03-12%20181146.png)

---

### 6. Derivatives + Eigenvalues

This tab performs symbolic analysis of a function in either:

- two variables: \(f(x,y)\)
- three variables: \(f(x,y,z)\)

It computes:

- the differential \(df\)
- partial derivatives
- the gradient
- the Hessian matrix
- Hessian eigenvalues
- evaluated values at a chosen point

This tab is useful for connecting symbolic derivative work with the geometric ideas shown in the other tabs.

![Alt text](https://github.com/VictorGranado/SDL-4-Optimization-Visualizer/blob/400de97f20c49ae5d9386cd8fb7fe9135b2e52a7/Screenshot%202026-03-12%20181155.png)

---

## Features

- interactive GUI built with Tkinter
- contour and surface visualization
- gradient and directional derivative tools
- critical point detection and classification
- Lagrange multiplier solver and plotter
- symbolic derivative, Hessian, and eigenvalue analysis
- quick example buttons for testing common functions

---

## Installation

### Requirements
- Python 3.10+
- NumPy
- Matplotlib
- SymPy

Install the required libraries with:

```bash
pip install numpy matplotlib sympy
