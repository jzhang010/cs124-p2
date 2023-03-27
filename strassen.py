import numpy as np
import sys
import time

n0 = 15
dim = int(sys.argv[2])
f = open(sys.argv[3], 'r') 

m1 = np.empty([dim, dim])
m2 = np.empty([dim, dim])

for i in range(dim):
  for j in range(dim): 
    m1[i][j] = f.readline() 

for i in range(dim):
  for j in range(dim): 
    m2[i][j] = f.readline() 

def strassen (m, n, d): 
  if d == 1: 
    return np.array(m*n)
  else: 
    if d % 2 == 1: 
      m = np.pad(m, pad_width=((0, 1), (0, 1)))
      n = np.pad(n, pad_width=((0, 1), (0, 1)))
    new_d = (d+1)//2
    A = m[:new_d,:new_d]
    B = m[:new_d,new_d:]
    C = m[new_d:,:new_d]
    D = m[new_d:,new_d:]
    E = n[:new_d,:new_d]
    F = n[:new_d,new_d:]
    G = n[new_d:,:new_d]
    H = n[new_d:,new_d:]
    P1 = strassen(A, F-H, new_d)
    P2 = strassen(A+B, H, new_d)
    P3 = strassen(C+D, E, new_d)
    P4 = strassen(D, G-E, new_d)
    P5 = strassen(A+D, E+H, new_d)
    P6 = strassen(B-D, G+H, new_d)
    P7 = strassen(C-A, E+F, new_d)
    return np.block([[-P2+P4+P5+P6, P1+P2], [P3+P4, P1-P3+P5+P7]])[:d,:d]

def standard_mult(m, n, d): 
  mn = np.zeros([d, d])
  for i in range(d): 
    for j in range(d):
      for k in range (d):
        mn[i][j] += m[i][k] * n[k][j]
  return mn

def test(n): 
  mat1 = np.random.randint(3, size=(n, n))
  mat2 = np.random.randint(3, size=(n, n))
  start = time.time()
  standard_mult(mat1, mat2, n)
  print(time.time() - start)
  start = time.time()
  strassen(mat1, mat2, n)
  print(time.time() - start)

# print(standard_mult(m1, m2, dim))
# print(strassen(m1, m2, dim))
#test(200)

print(m1)
m1.resize(4, 4)
print(m1)