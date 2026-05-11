import numpy as np

def calcular_energia_total(s, A):
    
    # np.outer(s, s) genera la matriz de productos s_i * s_j
    # np.triu(A, k=1) extrae solo la parte triangular superior (las aristas J_ij sin duplicar)
    energia_interacciones = -np.sum(np.triu(A, k=1) * np.outer(s, s))
    
    energia_local = -np.sum(np.diag(A) * s)
    
    return energia_interacciones + energia_local

def intento_transicion_local(s, A, t, energia_actual):
    
    N = len(s)
    
    i = np.random.randint(0, N)
    
    # Calcular delta E usando la fórmula optimizada del informe
    # A[i, :] es la fila i (interacciones con otros vértices)
    # np.dot(A[i, :], s) es la suma (incluyendo el término A_ii * s_i, por lo que 
    # ajustamos sacando s_i y multiplicando por el s_i actual (antes del flip))

    suma_vecinos = np.dot(A[i, :], s) - A[i, i] * s[i]
    delta_E = 2 * s[i] * (A[i, i] + suma_vecinos)
    
    aceptado = False
    if delta_E <= 0:
        aceptado = True
    else:
        # Criterio de Metrópolis
        r = np.random.uniform(0, 1)
        if r <= np.exp(-delta_E / t):
            aceptado = True
            
    if aceptado:
        s[i] *= -1 # Invertir el espín
        energia_actual += delta_E
        
    return s, energia_actual, aceptado

def simulated_annealing(A, t_0, t_f, L, alpha):
   
    N = A.shape[0]
    s = np.random.choice([-1, 1], size=N) # Configuración inicial aleatoria
    energia_actual = calcular_energia_total(s, A)
    
    s_star = np.copy(s)
    energia_star = energia_actual
    
    t = t_0
    evaluaciones = 0
    
    while t >= t_f:
        for k in range(L):
            s, energia_actual, _ = intento_transicion_local(s, A, t, energia_actual)
            evaluaciones += 1
            
            if energia_actual < energia_star:
                energia_star = energia_actual
                s_star = np.copy(s)
                
        # Enfriamiento geométrico
        t *= alpha
        
    return s_star, energia_star, evaluaciones

def parallel_tempering(A, temperaturas, C, L):
    
    N = A.shape[0]
    M = len(temperaturas)
    

    s_replicas = [np.random.choice([-1, 1], size=N) for _ in range(M)]
    energias = [calcular_energia_total(s_replicas[i], A) for i in range(M)]
    
    # Encontrar la mejor inicial
    idx_mejor = np.argmin(energias)
    s_star = np.copy(s_replicas[idx_mejor])
    energia_star = energias[idx_mejor]
    
    evaluaciones_totales = 0
    
    for c in range(C):
        # Paso 1.1: Búsqueda intra-cadena (L iteraciones por réplica)
        for i in range(M):
            t_i = temperaturas[i]
            for _ in range(L):
                s_replicas[i], energias[i], _ = intento_transicion_local(s_replicas[i], A, t_i, energias[i])
                evaluaciones_totales += 1
                
                # Actualizar óptimo global
                if energias[i] < energia_star:
                    energia_star = energias[i]
                    s_star = np.copy(s_replicas[i])
                    
        # Paso 1.2: Intercambios inter-cadena (Swapping)
        for i in range(M - 1):
            delta_beta = (1.0 / temperaturas[i]) - (1.0 / temperaturas[i+1])
            delta_E = energias[i] - energias[i+1]
            
            # alfa de aceptación para el swap
            # delta_beta siempre es > 0 porque T_i < T_{i+1}. 
            # Si E_i > E_{i+1}, exp es positivo -> aceptamos (baja la energía hacia T más fría)
            exponente = delta_beta * delta_E
            
            # Prevenir overflow en np.exp
            if exponente > 0:
                alfa = 1.0 
            else:
                alfa = np.exp(exponente)
                
            r = np.random.uniform(0, 1)
            if r <= alfa:
                # Intercambiar las configuraciones y sus energías
                s_replicas[i], s_replicas[i+1] = s_replicas[i+1], s_replicas[i]
                energias[i], energias[i+1] = energias[i+1], energias[i]
                
    return s_star, energia_star, evaluaciones_totales