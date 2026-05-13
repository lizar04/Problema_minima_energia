import networkx as nx
import numpy as np
import os

def generar_matriz_A(N, sigma_J=1.0, sigma_H=1.0, semilla=None):
    
    if semilla is not None:
        np.random.seed(semilla)
        
    if N % 2 != 0:
        raise ValueError("El número de nodos N debe ser par.")
        
    G = nx.random_regular_graph(d=3, n=N, seed=semilla)
    
    A = np.zeros((N, N))
    
    for i in range(N):
        A[i, i] = np.random.normal(0, sigma_H)
        
    for u, v in G.edges():
        peso_J = np.random.normal(0, sigma_J)
        A[u, v] = peso_J
        A[v, u] = peso_J 
        
    return A

os.makedirs("instancias_espin", exist_ok=True)


dimensiones = {
    "pequeno": [10, 20],        # Para probar el modelo exacto PLE
    "mediano": [50, 100],       # Para calibración de parámetros de SA y PT
    "grande": [200, 500]        # Para probar escalabilidad y estancamiento
}


for categoria, lista_N in dimensiones.items():
    for N in lista_N:
        for inst in range(1, 6):
            semilla_instancia = 42 + N + inst
            
            # sigma_H=0.0 asegura la frustración máxima
            matriz_A = generar_matriz_A(N, sigma_J=1.0, sigma_H=1.0, semilla=semilla_instancia)
            
            nombre_archivo = f"instancias_espin/matriz_A_{categoria}_N{N}_instancia{inst}.npy"
            np.save(nombre_archivo, matriz_A)

