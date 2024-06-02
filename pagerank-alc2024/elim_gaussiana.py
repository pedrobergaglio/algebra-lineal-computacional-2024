#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliminacion Gausianna
"""
import numpy as np

def elim_gaussiana(A):
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    ## desde aqui -- CODIGO A COMPLETAR
    print('Matriz A \n', Ac)

    for n in range(0, m-1):

        pivot = Ac[n,n]
        print("Pivot:", pivot)

        print("Estamos trabajando en: \n", Ac[n:,n:])

        for i in range(n+1,m):

            
            #print("Casilla i, n:", Ac[i,n])
            
            cociente = Ac[i, n]/pivot
            
            print("cociente:", cociente)

            #print("Fila i:", Ac[i, :])
            #print("Fila n:", Ac[n, :])

            Ac[i, n:] = Ac[i, n:] - cociente*Ac[n, n:]

            Ac[i, n] = cociente

            cant_op += 1

        print(f'Matriz A ({n+1}) \n', Ac)




                
    ## hasta aqui
            
    L = np.tril(Ac,-1) + np.eye(A.shape[0]) 
    U = np.triu(Ac)
    
    return L, U, cant_op


def main():
    n = 7
    B = np.eye(n) - np.tril(np.ones((n,n)),-1) 
    B[:n,n-1] = 1
    print('Matriz B \n', B)
    
    L,U,cant_oper = elim_gaussiana(B)
    
    print('Matriz L \n', L)
    print('Matriz U \n', U)
    print('Cantidad de operaciones: ', cant_oper)
    print('B=LU? ' , 'Si!' if np.allclose(np.linalg.norm(B - L@U, 1), 0) else 'No!')
    print('Norma infinito de U: ', np.max(np.sum(np.abs(U), axis=1)) )

if __name__ == "__main__":
    main()
    
    