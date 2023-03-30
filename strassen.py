import numpy as np
import sys
import time
import random

# Reads command line inputs
n0 = 29
flag = int(sys.argv[1])
dim = int(sys.argv[2])
f = open(sys.argv[3], 'r') 

# Initializes and reads in input matrix elements
m1 = np.empty([dim, dim], dtype=np.int32)
m2 = np.empty([dim, dim], dtype=np.int32)

for i in range(dim):
  for j in range(dim): 
    m1[i][j] = f.readline() 

for i in range(dim):
  for j in range(dim): 
    m2[i][j] = f.readline() 

# Calculates product of two matrices using Strassen's
def strassen (m, n, d): 
  if d <= n0: 
    return standard_mult(m, n, d)
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

# Calculates product of two matrices using standard matrix multiplcation 
def standard_mult(m, n, d): 
  mn = np.zeros([d, d], dtype=np.int32)
  for i in range(d): 
    for j in range(d):
      for k in range (d):
        mn[i][j] += m[i][k] * n[k][j]
  return mn

# Generates random graph and returns number of triangles
def gen_graph(p):
  adj = np.zeros([1024, 1024])
  for i in range(1023):
    for j in range(i + 1, 1024): 
      if random.uniform(0, 1) < p: 
        adj[i][j] = 1
        adj[j][i] = 1
  start = time.time()
  a2 = strassen(adj, adj, 1024)
  print(time.time() - start)
  a3 = strassen(a2, adj, 1024)
  print(time.time() - start)
  tot = 0
  for i in range(1024):
    tot += a3[i][i]
  return tot / 6

# Gives the number of triangles for all of the different graphs
def calc_triangles():
  for i in range (1, 6):
    print(gen_graph(i*.01))

# Returns if strassen is optimal for a range of dimensions 
def find_n0(): 
  global n0 
  for n in range (1, 50): 
    mat1 = np.random.randint(2, size=(n, n))
    mat2 = np.random.randint(2, size=(n, n))
    stand_avg = 0
    strat_avg = 0
    strat_opt = False
    for _ in range (5):
      start = time.time()
      standard_mult(mat1, mat2, n)
      stand_avg += time.time() - start
    stand_avg /= 5
    min = stand_avg
    n0 = (n + 1) // 2
    while n0 > 1: 
      strat_avg = 0
      for _ in range (5): 
        start = time.time()
        strassen(mat1, mat2, n)
        strat_avg += time.time() - start
      strat_avg /= 5
      if strat_avg > min and min != stand_avg:
        break
      if (strat_avg < min):
        strat_opt = True
        break 
      n0 = (n0 + 1) // 2
    print("n: %d, strassen optimal: " % (n), strat_opt)

# Finds the optimal range for n0 given n
def test(n): 
  global n0
  mat1 = np.random.randint(2, size=(n, n))
  mat2 = np.random.randint(2, size=(n, n))
  start = time.time()
  standard_mult(mat1, mat2, n)
  t1 = time.time() - start
  opt = n
  best = -1
  n0 = n
  while n0 > 0: 
    n0 = (n0 + 1) // 2
    start = time.time()
    strassen(mat1, mat2, n)
    t2 = time.time() - start
    print("n0: %d, standard: %f, strassen: %f, " % (n0, t1, t2), t2 <= t1)
    if best != -1 and t2 > best or n0 == 1: 
      break 
    if t2 <= t1 and (best == -1 or t2 < best): 
      best = t2
      opt = n0
  return opt

#print(standard_mult(m1, m2, dim))
ans = strassen(m1, m2, dim)

# Print output of matrix product
if flag == 0:
  for i in range(dim): 
    print(ans[i][i])

if flag == 1: 
  i = 2
  while i <= 1024: 
    print("size: %d, optimal: " % (i), test(i))
    i *= 2
  i = 3
  while i <= 1024: 
    print("size: %d, optimal: " % (i), test(i))
    i *= 3
  i = 5
  while i <= 1024: 
    print("size: %d, optimal: " % (i), test(i))
    i *= 5

if flag == 2: 
  find_n0()

if flag == 3: 
  calc_triangles()