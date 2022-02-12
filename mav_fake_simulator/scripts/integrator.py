import numpy as np


def my_RK2(x, u, f, h, p):
    # Perform one step explicit RK2 integrator for nonlinear system x_dot = f(x, u, p)
    # Inputs:
    #   x: current state, array, (nx,)
    #   u: current control input, array, (nu,)
    #   f: nonlinear system dynamics function
    #   h: step size (dt)
    #   p: passing parameters (to f), (np,)
    # Outputs:
    #   x_next: next step state after integration, array
    k1 = h * f(x, u, p)
    k2 = h * f(x + 0.5 * k1, u, p)
    x_next = x + k2

    return x_next


def my_RK4(x, u, f, h, p):
    # Perform one step explicit RK4 integrator for nonlinear system x_dot = f(x, u, p)
    # Inputs:
    #   x: current state, array, (nx,)
    #   u: current control input, array, (nu,)
    #   f: nonlinear system dynamics function
    #   h: step size (dt)
    #   p: passing parameters (to f), (np,)
    # Outputs:
    #   x_next: next step state after integration, array
    k1 = h * f(x, u, p)
    k2 = h * f(x + 0.5 * k1, u, p)
    k3 = h * f(x + 0.5 * k2, u, p)
    k4 = h * f(x + k3, u, p)
    x_next = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    return x_next
