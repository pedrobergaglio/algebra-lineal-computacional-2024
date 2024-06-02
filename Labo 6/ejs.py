"""
Hacer un programa que reciba una matriz A ∈ R
n×n y un entero positivo k y que aplique k
iteraciones del m´etodo de la potencia con un vector aleatorio inicial v ∈ R
n. El programa debe devolver
un vector a ∈ R
k
, donde ai sea la aproximaci´on al autovalor obtenida en el paso i.
"""

import numpy as np
import matplotlib.pyplot as plt

def power_method(A, k):
    n = A.shape[0]
    v = np.random.random(n)
    historical_values = np.zeros(k)

    for i in range(k):
        
        v = np.dot(A, v)
        historical_values[i] = np.linalg.norm(v, 2)

        v = v / np.linalg.norm(v)
        
        #print(v)

    eigenvals = np.linalg.eigvals(A)
    #order eigenvals and print the first 2
    #print(np.sort(eigenvals)[:2])
        
    lambda_max = np.linalg.norm(np.linalg.eigvals(A), np.inf)
    print (np.abs(np.sort(-eigenvals)[0]))
    print("lambda max", lambda_max)

    lambda_max_arr = np.full(k, lambda_max)

    error = np.log(np.abs(historical_values - lambda_max_arr))
    print(error[:3])
    print(historical_values[:3])

    """plt.subplot(1, 2, 1)
    plt.plot(range(1, k+1), error)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    for i, err in enumerate(error):
        if i % 33 == 0:
            plt.annotate(f'({i+1}, {err:.2f})', (i+1, err))

    plt.subplot(1, 2, 2)"""
    plt.plot(range(k), error)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    for i, err in enumerate(error):
        if i % 20 == 0:
            plt.annotate(f'({i+1}, {err:.2f})', (i+1, err))
    # plot also function y(x) = 2 log(λ2/λ1)x + log(e)
    log_e = np.log(error)
    log_lambda1y2 = 2*np.log(np.abs(np.sort(-eigenvals)[1])/lambda_max)
    log_lambda1y2_porx = log_lambda1y2 * np.arange(k)
    
    plt.plot(range(k), log_lambda1y2_porx + log_e, label='2 log(λ2/λ1)x + log(e)')
    plt.plot(range(k), log_lambda1y2_porx, label='2 log(λ2/λ1)x')

    plt.show()

#A = np.array([[1, 2], [3, 4]])

#create a random 100x100 matrix
C = np.random.random((100, 100))
#A  = np.eye(100) * 2 + np.eye(100, k=1) + np.eye((100), k=-1)*100
#print(A)

#power_method(A, 100)

A = 0.5*(C + C.T)
B = A + np.eye(100) * 500

power_method(B, 100)