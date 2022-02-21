import numpy as np
from numpy import array
from random import gauss

# N * N points
def gen_points(N, A, B, C, mu = 0, sigma = 0):
  res = []
  for i in range(0, N):
    for j in range(0, N): #Ax + By + Cz = 1
      u = i/N
      v = j/N
      noise = gauss(mu, sigma)

      if np.abs(A) > 0.001:
        x = noise + (1 - B * u - C * v)/A 
        res.append([x, u, v])
      elif np.abs(B) > 0.001:
        x = noise + (1 - A * u - C * v)/B
        res.append([u, x, v])
      elif np.abs(C) > 0.001:
        x = noise + (1 - A * u - B * v)/C
        res.append([u, v, x])
      else:
        raise("Degenerate case")

  return res

def regression(points):
  xs = points[:, 0]
  ys = points[:, 1]
  zs = points[:, 2]
  N = points.shape[0]
  
  sumx = sum(xs)
  sumy = sum(ys)
  sumz = sum(zs)
  
  a = sum(xs * xs)
  b = sum(ys * ys)
  c = sum(zs * zs)
  
  p = sum(xs * ys)
  q = sum(xs * zs)
  r = sum(ys * zs)

  m = array([[a, p, q], [p, b, r], [q, r, c]])
  
  #print(m)

  inv_mat = np.linalg.inv(m)

  return np.dot(inv_mat, array([sumx, sumy, sumz]))  

def sse(points, plane):
  s = 0
  for i in range(0, points.shape[0]):
    d = np.dot(points[i], plane) - 1
    s = s + d * d
  return s

points = array(gen_points(5, 1, 1, 1))