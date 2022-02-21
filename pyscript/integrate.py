from __future__ import division
from sympy import *
init_printing(use_unicode = True)

x, theta, phi_n, a = symbols('x theta phi_n a')

D = cos(x)/(1 + (a**2 - 1)/(2 + 2 * sin(x)) * (cos(x) * cos(theta) * cos(phi_n) + (sin(x) + 1) * sin(phi_n))**2)**2