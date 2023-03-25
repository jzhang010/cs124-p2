import numpy as np

dim = 3
n0 = 15

m1 = np.array([[1, 0, 2], [0, 1, 3], [0, 1, 3]])
m2 = np.array([[4, 3, 4], [2, 4, 1], [0, 1, 3]])

def strassen (m, n, d): 
  if d == 1: 
    return np.array(m*n)
  else: 
    if d % 2 == 1: 
      m = np.pad(m, pad_width=((0, 1), (0, 1)))
      n = np.pad(n, pad_width=((0, 1), (0, 1)))
    new_d = (d+1)//2
    A = m[0:new_d,0:new_d]
    B = m[0:new_d,new_d:new_d*2]
    C = m[new_d:new_d*2,0:new_d]
    D = m[new_d:new_d*2,new_d:new_d*2]
    E = n[0:new_d,0:new_d]
    F = n[0:new_d,new_d:new_d*2]
    G = n[new_d:new_d*2,0:new_d]
    H = n[new_d:new_d*2,new_d:new_d*2]
    P1 = strassen(A, F-H, new_d)
    P2 = strassen(A+B, H, new_d)
    P3 = strassen(C+D, E, new_d)
    P4 = strassen(D, G-E, new_d)
    P5 = strassen(A+D, E+H, new_d)
    P6 = strassen(B-D, G+H, new_d)
    P7 = strassen(C-A, E+F, new_d)
    return np.block([[-P2+P4+P5+P6, P1+P2], [P3+P4, P1-P3+P5+P7]])[0:d,0:d]

def standard_matrix(m, n, d): 
  mn = np.zeros([d, d])
  for i in range(d): 
    for j in range(d):
      for k in range (d):
        mn[i][j] += m[i][k] * n[k][j]
  return mn

print(standard_matrix(m1, m2, 3))
print(strassen(m1, m2, 3))