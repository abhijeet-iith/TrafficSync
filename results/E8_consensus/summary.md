# E8 - consensus validation

Path graph, x(0) = [2, 5, 8]: x(100) = [5.0, 5.0, 5.0], max |1'x - 15| over iterations = 4.6e-14.

Ring of 8, eps = f x 2/lambda_max: spread after 400 iterations f=0.5: 3.6e-15, f=0.9: 8.9e-16, f=0.99: 3.1e-04, f=1.01: 2.8e+03, f=1.1: 4.7e+31

Erdos-Renyi G(20, p), 24 connected graphs: Spearman(lambda2, iterations to 1e-6) = -0.99.

Tracking a drifting average on grid 3x3 (K = 5 rounds per step): dynamic consensus error 0.0013, restarted static 0.0120.
