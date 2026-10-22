import numpy as numpy

obstaculos = [
    poligono_regular((3,3), 4, 0.8),
    poligono_regular((7,5), 3, 1.0),
    poligono_regular((5,8), 6, 1.2),
    ] 

def poligono_regular(centro, n, raio, rotacao=0.0):
    angulos = np.linspace(0, 2*np.pi, n, endpoint=False) + rotacao
    return np.column_stack([
        centro[0] + radius*np.cos(angulos), 
        centro[1] + radius*np.sin(angulos)
    ])

def potencial(q, objetivo, obstaculos, k_attr=1.0, k_rep=100.0, d0=1.5):
    d_objetivo = np.linalg.norm(q - objetivo)
    U = 0.5 * k_attr * (d_objetivo**2)
    grad = k_attr * (q - objetivo)

    for poligono in obstaculos:
        d = dist_ponto_poligono(q, poligono)
        if d<= d0:
            U += 0.5 * k_rep * (1.0/d - 1.0/d0)**2
            direcao = (q - ponto_mais_proximo_poligono(q, poly))
            direcao /= (np.linalg.norm(direction) + 1e-9)
            grad += k_rep * (1.0/d - 1.0/d0) * (1.0/d**2) * direction
    return U, grad

def plan_potencial(inicio, objetivo, obstaculos, alpha = 0.05, max_iter=1000):
    q = np.array(inicio, dtype=float)
    path = [q.copy()]
    for i in range(max_iter):
        U, grad = potencial(q, objetivo, obstaculos)
        if np.linalg.norm(grad) < 1e-3:
            break
        q = q - alpha * grad
        path.append(q.copy())
        if np.linalg.norm(q - goal) < 0.1:
            break
    return np.array(path)


