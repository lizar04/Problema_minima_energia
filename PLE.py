import numpy as np
import pulp
import time

def resolver_ple_exacto(matriz_A):
    
    N = matriz_A.shape[0]
    
    start_time = time.time()
    
    prob = pulp.LpProblem("Minima_Energia_Espin", pulp.LpMinimize)
    
    # x_i in {0, 1} para los nodos
    x = pulp.LpVariable.dicts("x", range(N), cat=pulp.LpBinary)
    
    # Identificar las aristas (solo miramos la parte triangular superior para no duplicar)
    aristas = []
    for i in range(N):
        for j in range(i + 1, N):
            if matriz_A[i, j] != 0:
                aristas.append((i, j))
                
    # y_ij in {0, 1} para la linealización de x_i * x_j
    y = pulp.LpVariable.dicts("y", aristas, cat=pulp.LpBinary)
    
    
    funcion_objetivo = []
    
    # Término de las interacciones (aristas)
    for (i, j) in aristas:
        J_ij = matriz_A[i, j]
        # -J_ij * (4*y_ij - 2*x_i - 2*x_j + 1)
        termino_arista = -J_ij * (4*y[i, j] - 2*x[i] - 2*x[j] + 1)
        funcion_objetivo.append(termino_arista)
        
    # Término de las influencias locales (nodos)
    for i in range(N):
        H_i = matriz_A[i, i]
        # -H_i * (2*x_i - 1)
        termino_nodo = -H_i * (2*x[i] - 1)
        funcion_objetivo.append(termino_nodo)
        
    prob += pulp.lpSum(funcion_objetivo), "Energia_Total"
    
    # restricciones (Linealización)
    for (i, j) in aristas:
        prob += y[i, j] <= x[i], f"Restriccion_y1_{i}_{j}"
        prob += y[i, j] <= x[j], f"Restriccion_y2_{i}_{j}"
        prob += y[i, j] >= x[i] + x[j] - 1, f"Restriccion_y3_{i}_{j}"
        

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    tiempo_ejecucion = time.time() - start_time
    
    # Mapear de vuelta de x_i (0 o 1) a s_i (-1 o 1)
    configuracion_optima = np.zeros(N)
    for i in range(N):
        valor_x = x[i].varValue
        configuracion_optima[i] = 2 * valor_x - 1
        
    energia_minima = pulp.value(prob.objective)
    
    return configuracion_optima, energia_minima, tiempo_ejecucion


if __name__ == "__main__":
    try:
        A_prueba = np.load("instancias_espin/matriz_A_pequeno_N10.npy")
        
        s_optimo, E_min, tiempo = resolver_ple_exacto(A_prueba)
        
        print("RESULTADOS DEL PLE")
        print(f"Energía mínima encontrada: {E_min:.4f}")
        print(f"Configuración óptima (s): {s_optimo}")
        print(f"Tiempo de ejecución: {tiempo:.4f} segundos")
        
    except FileNotFoundError:
        print("genera la matriz A")