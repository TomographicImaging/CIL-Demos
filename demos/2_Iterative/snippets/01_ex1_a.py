alpha = 30
G = alpha*L1Norm()
F = LeastSquares(A, b)

myFISTAL1 = FISTA(f=F, 
                  g=G, 
                  initial=x0, 
                  max_iteration=1000, 
                  update_objective_interval=10)