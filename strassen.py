import numpy as np
import sys
import time
import random

n0 = 97
flag = int(sys.argv[1])
dim = int(sys.argv[2])
f = open(sys.argv[3], 'r') 

m1 = np.empty([dim, dim], dtype=np.int32)
m2 = np.empty([dim, dim], dtype=np.int32)

for i in range(dim):
  for j in range(dim): 
    m1[i][j] = f.readline() 

for i in range(dim):
  for j in range(dim): 
    m2[i][j] = f.readline() 

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

def standard_mult(m, n, d): 
  mn = np.zeros([d, d], dtype=np.int32)
  for i in range(d): 
    for j in range(d):
      for k in range (d):
        mn[i][j] += m[i][k] * n[k][j]
  return mn

def find_n0(beg, end, l, h): 
  global n0 
  for n0 in range (beg, end + 1): 
    n = n0 + 1
    mat1 = np.random.randint(low=l,high=h, size=(n, n))
    mat2 = np.random.randint(low=l,high=h, size=(n, n))
    stand_avg = 0
    strat_avg = 0
    for _ in range (5):
      start = time.time()
      standard_mult(mat1, mat2, n)
      stand_avg += time.time() - start
      start = time.time()
      strassen(mat1, mat2, n)
      strat_avg += time.time() - start
    stand_avg /= 5
    strat_avg /= 5
    print("n0: %d, standard: %f, strassen: %f, " % (n0, stand_avg, strat_avg), strat_avg <= stand_avg)
    n += 1

def gen_graph(p):
  adj = np.zeros([1024, 1024])
  for i in range(1023):
    for j in range(i + 1, 1024): 
      if random.uniform(0, 1) < p: 
        adj[i][j] = 1
        adj[j][i] = 1
  print(adj)
  start = time.time()
  a2 = strassen(adj, adj, 1024)
  print(time.time() - start)
  a3 = strassen(a2, adj, 1024)
  print(time.time() - start)
  tot = 0
  for i in range(1024):
    tot += a3[i][i]
  return tot / 6


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

def calc_triangles():
  for i in range (1, 6):
    print(gen_graph(i*.01))

#print(standard_mult(m1, m2, dim))
ans = strassen(m1, m2, dim)

if flag == 0:
  for i in range(dim): 
    print(ans[i][i])

if flag == 1:   
  print("0, 1 matrices")
  find_n0(1, 100, 0, 2)
  print("0, 1, 2 matrices")
  find_n0(1, 100, 0, 3)
  print("-1, 0, 1 matrices")
  find_n0(1, 100, -1, 1)

if flag == 2: 
  calc_triangles()

if flag == 3: 
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
  i = 25
  while i <= 1024: 
    print("size: %d, optimal: " % (i), test(i))
    i = i*2 - 1