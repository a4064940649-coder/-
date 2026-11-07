import sys, time; sys.path.insert(0, sys.argv[1])
import rlib as R
t=time.time()
x=R.paths(2026110701,'MA3',8,4,2,0,200000); print('MA3|10.4|0.78 q', R.bstar(x), time.time()-t)
x=R.paths(2026110702,'MA3',7,4,2,0,100000); print(R.evaluate(x))
x=R.paths(2026110702,'D3',7,4,2,0,100000); print(R.evaluate(x), time.time()-t)
