def o_sign_nz(k):
  if k >= 0.0:
    return 1.0
  else:
    return -1.0

def oct_encode(x, y, z):
  l1 = abs(x) + abs(y) + abs(z)
  nx = x/l1
  ny = y/l1
  if z < 0.0:
    nx = (1 - abs(ny)) * o_sign_nz(nx)
    ny = (1 - abs(nx)) * o_sign_nz(ny)
  return 0.5 * nx + 0.5, 0.5 * ny + 0.5 