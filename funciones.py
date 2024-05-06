import numpy as np
import networkx as nx
import scipy.linalg as la
import time


def leer_archivo(input_file_path):

    f = open(input_file_path, 'r')
    n = int(f.readline())
    m = int(f.readline())
    W = np.zeros(shape=(n,n))
    for _ in range(m):
        line = f.readline()
        i = int(line.split()[0]) - 1
        j = int(line.split()[1]) - 1
        W[j,i] = 1.0
    f.close()
    
    return W

def medir_tiempo_ejecucion(W, p):
    start_time = time.time()
    calcularRanking(W, p)
    end_time = time.time()
    return end_time - start_time

def dibujarGrafo(W, print_ejes=False):
    
    options = {
    'node_color': 'yellow',
    'node_size': 200,
    'width': 3,
    'arrowstyle': '-|>',
    'arrowsize': 10,
    'with_labels' : True}
    
    N = W.shape[0]
    G = nx.DiGraph(W.T)
    
    #renombro nodos de 1 a N
    G = nx.relabel_nodes(G, {i:i+1 for i in range(N)})
    if print_ejes:
        print('Ejes: ', [e for e in G.edges])
    
    nx.draw(G, pos=nx.spring_layout(G), **options)

def calcularRanking(W, p):
    npages = W.shape[0]
    rnk = np.arange(0, npages) # ind[k] = i, la pagina k tienen el iesimo orden en la lista.
    scr = np.zeros(npages) # scr[k] = alpha, la pagina k tiene un score de alpha 

    #calculo D
    D = np.zeros((npages,npages))
    for i in range(npages):
        if np.sum(W[i]) == 0:
            D[i][i] = 0
        else:
            D[i][i] = 1/np.sum(W[i])
    print(D)
    print(p*(W@D))

    #armo la matriz M
    M = np.eye(npages) - p*(W@D)

    print(M)

    #calculo L y U
    L, U = factorizarLU(M)

    #calculo el score con el sistema de ecuaciones
    e = np.transpose(np.ones(npages)) # vector de unos, la salida del sistema de ecuaciones
    y = la.solve_triangular(L, e, lower=True, check_finite=False) # resuelvo Ly=e
    scr = la.solve_triangular(U, y, check_finite=False) # resuelvo Ux=y

    # normalizo el score
    scr = scr / np.sum(scr)

    return rnk, scr

def factorizarLU(A):

    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    #print('Matriz A \n', Ac)

    for n in range(0, m-1): # n es la columna en la que estamos trabajando

        pivot = Ac[n,n]
        #print("Pivot:", pivot)
        #print("Estamos trabajando en: \n", Ac[n:,n:])

        for i in range(n+1,m): # i es la fila en la que estamos trabajando
            
            #print("Casilla i, n:", Ac[i,n])
            cociente = Ac[i, n]/pivot # cociente es el valor que se va a restar a la fila i
            
            #print("cociente:", cociente)
            #print("Fila i:", Ac[i, :])
            #print("Fila n:", Ac[n, :])

            # actualizo la fila i
            # sólo modificamos los valores de la fila i que están a la derecha de la columna n
            Ac[i, n:] = Ac[i, n:] - cociente*Ac[n, n:]

            # actualizo la casilla i,n con la modificación que le hicimos a la fila i
            Ac[i, n] = cociente

            cant_op += 1 # sumo una operación por cada casilla que modificamos

        #print(f'Matriz A ({n+1}) \n', Ac)
    
    # obtengo L y U a partir de la matriz Ac modificada por eliminación gaussiana
    L = np.tril(Ac,-1) + np.eye(A.shape[0]) # matriz triangular inferior con unos en la diagonal
    U = np.triu(Ac) # matriz triangular superior
    
    return L, U #, cant_op


def obtenerMaximoRankingScore(M, p):
    output = -np.inf
    # calculo el ranking y los scores
    rnk, scr = calcularRanking(M, p)
    output = np.max(scr)
    
    return output


W = leer_archivo('tests/test_dosestrellas.txt')

#dibujarGrafo(W)
#print(calcularRanking(W, 0.5))

W = np.array([  [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0]])

print(obtenerMaximoRankingScore(W, 0.9))