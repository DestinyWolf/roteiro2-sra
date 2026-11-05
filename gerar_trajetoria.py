"""Geracao de trajetoria com Campos de Potencial em Grid (Brushfire + A*) e Minkowski.

Le o mapa de obstaculos poligonais produzido por 'gerar_mapa.py'
(mapa_obstaculos.json) e gera uma trajetoria segura do ponto inicial
ate o objetivo.

Modelagem e algoritmos implementados:
1. Soma de Minkowski ('expandir_obstaculos'): infla cada obstaculo (e
   encolhe o contorno da mesa) pelo raio do robo ('--raio-robo' + '--margem'),
   permitindo tratar o robo como uma particula pontual no espaco de
   configuracoes (C-space).
2. Decomposicao em Grid ('construir_grid'): discretiza o C-space em
   celulas regulares livres ou ocupadas.
3. Algoritmo Brushfire ('mapa_brushfire'): espalhamento em largura (BFS)
   a partir das celulas ocupadas e limites da mesa para calcular a
   distancia de cada celula livre ate o obstaculo mais proximo.
4. Campo de Potencial no Grid ('potencial_repulsivo_brushfire'): gera um
   potencial repulsivo nas celulas proximas aos obstaculos (dentro do
   raio de influencia '--d0'), empurrando o caminho para longe das quinas
   e paredes para compensar a deriva da odometria.
5. Busca A* ('busca_a_estrela'): encontra o caminho de menor custo
   combinando a distancia euclidiana/octil (potencial atrativo ao objetivo)
   com o potencial repulsivo do Brushfire.
6. Suavizacao e Limitacao de Waypoints ('suavizar_caminho', 'reduzir_pontos'):
   remove degraus desnecessarios da grade preservando a folga do Brushfire
   e garante no maximo 15 waypoints (16 pontos contando o inicio), sem
   subdividir retas (evitando paradas intermediarias que causam deriva
   no robo de auto-equilibrio).
7. Comunicacao com o Robo (protocolo de 'demo.py'): envia a pose inicial
   com (x, y, theta_inicial) via <O,x,y,th> e os waypoints com (x, y, 0.0)
   via <W,...> / <A,x,y,0.0>, com pre-compensacao de escala e deriva
   da odometria interna (--fator-odometria, --comp-angular-graus).
"""

import argparse
import heapq
import json
import numpy as np
import limites_mesa
import serial
import serial.tools.list_ports
import sys
import time

DEFAULT_PORT = "/dev/ttyUSB1"
DEFAULT_BAUD = 57600
TIMEOUT_S = 1.0
RESPONSE_TIMEOUT_S = 2.0
AUTO_EQ_S = 2.0
START_GRACE_S = 2.5

STATE_NAMES = {0: "GIRAR", 1: "ANDAR", 2: "DONE"}

# Restricao do professor: no maximo 15 waypoints alem do ponto inicial
# (total de 16 pontos na trajetoria contando o ponto de partida).
# MIN_PONTOS_TRAJETORIA = 0 evita subdividir segmentos retos em pontos
# colineares (cada parada intermediaria no Minibalance faz o controlador
# alternar ANDAR -> GIRAR -> ANDAR, gerando solavancos e deriva angular).
MAX_WAYPOINTS_ROBO = 15
MAX_PONTOS_TRAJETORIA = MAX_WAYPOINTS_ROBO + 1
MIN_PONTOS_TRAJETORIA = 0


# ---------------------------------------------------------------- geometria

def poligono_regular(centro, n, raio, rotacao=0.0):
    """Retorna os n vertices (n x 2) de um poligono regular."""
    angulos = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotacao
    return np.column_stack([
        centro[0] + raio * np.cos(angulos),
        centro[1] + raio * np.sin(angulos),
    ])


def ponto_mais_proximo_segmento(q, a, b):
    """Ponto do segmento ab mais proximo de q."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return a.copy()
    t = np.clip(float((q - a) @ ab) / denom, 0.0, 1.0)
    return a + t * ab


def ponto_mais_proximo_poligono(q, poligono):
    """Ponto da borda do poligono mais proximo de q."""
    melhor = None
    melhor_d = np.inf
    n = len(poligono)
    for i in range(n):
        a = poligono[i]
        b = poligono[(i + 1) % n]
        p = ponto_mais_proximo_segmento(q, a, b)
        d = np.linalg.norm(q - p)
        if d < melhor_d:
            melhor_d = d
            melhor = p
    return melhor


def ponto_dentro_poligono(q, poligono):
    """Teste de inclusao por ray casting."""
    dentro = False
    n = len(poligono)
    for i in range(n):
        x1, y1 = poligono[i]
        x2, y2 = poligono[(i + 1) % n]
        if (y1 > q[1]) != (y2 > q[1]):
            x_cruz = x1 + (q[1] - y1) * (x2 - x1) / (y2 - y1)
            if q[0] < x_cruz:
                dentro = not dentro
    return dentro


def dist_ponto_poligono(q, poligono):
    """Distancia de q a borda do poligono (negativa se q estiver dentro)."""
    p = ponto_mais_proximo_poligono(q, poligono)
    d = np.linalg.norm(q - p)
    return -d if ponto_dentro_poligono(q, poligono) else d


# ------------------------------------------------- expansao (Minkowski)

def expandir_obstaculos(obstaculos, margem, limites, passo=0.002):
    """Soma de Minkowski de cada obstaculo com um disco de raio 'margem'.

    Implementada como dilatacao morfologica com elemento estruturante
    circular: por definicao, a dilatacao de um conjunto por um disco E a
    soma de Minkowski dos dois. Trata o robo de raio 'margem' como uma
    particula pontual no espaco de configuracoes.
    """
    if margem <= 0.0:
        return [np.asarray(o, dtype=float) for o in obstaculos]

    import cv2

    x0, y0 = limites["x_min"], limites["y_min"]
    borda = margem + 4 * passo
    largura = int(np.ceil((limites["x_max"] - x0 + 2 * borda) / passo))
    altura = int(np.ceil((limites["y_max"] - y0 + 2 * borda) / passo))

    def para_pixel(pts):
        return np.round((np.asarray(pts) - [x0 - borda, y0 - borda])
                        / passo).astype(np.int32)

    mascara = np.zeros((altura, largura), dtype=np.uint8)
    for poligono in obstaculos:
        cv2.fillPoly(mascara, [para_pixel(poligono)], 255)

    raio_px = int(np.ceil(margem / passo))
    disco = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * raio_px + 1, 2 * raio_px + 1))
    mascara = cv2.dilate(mascara, disco)

    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    expandidos = []
    for c in contornos:
        c = cv2.approxPolyDP(c, 0.8, True)
        pts = c.reshape(-1, 2).astype(float) * passo + [x0 - borda,
                                                        y0 - borda]
        if len(pts) >= 3:
            expandidos.append(pts)
    return expandidos


def encolher_limites(limites, margem):
    """Workspace reduzido da mesma margem aplicada aos obstaculos."""
    if not limites or margem <= 0.0:
        return limites
    encolhidos = {
        "x_min": limites["x_min"] + margem,
        "x_max": limites["x_max"] - margem,
        "y_min": limites["y_min"] + margem,
        "y_max": limites["y_max"] - margem,
    }
    contorno = limites_mesa.contorno_de_limites(limites)
    if contorno is not None:
        encolhidos["contorno"] = limites_mesa.encolher_contorno(contorno, margem)
    return encolhidos


# ------------------------------------------------ potencial continuo (opcional)

def potencial(q, objetivo, obstaculos, k_attr=1.0, k_rep=0.001, d0=0.08,
              d_max=0.3, limites=None):
    """Potencial continuo total e seu gradiente em q."""
    q = np.asarray(q, dtype=float)
    objetivo = np.asarray(objetivo, dtype=float)

    erro = q - objetivo
    d_objetivo = np.linalg.norm(erro)
    if d_objetivo <= d_max:
        U = 0.5 * k_attr * d_objetivo ** 2
        grad = k_attr * erro
    else:
        U = k_attr * d_max * d_objetivo - 0.5 * k_attr * d_max ** 2
        grad = k_attr * d_max * erro / d_objetivo

    fontes = []
    for poligono in obstaculos:
        d = dist_ponto_poligono(q, poligono)
        if d < d0:
            p = ponto_mais_proximo_poligono(q, poligono)
            direcao = q - p
            norma = np.linalg.norm(direcao)
            if norma > 1e-9:
                fontes.append((d, direcao / norma))

    if limites:
        paredes = [
            (q[0] - limites["x_min"], np.array([1.0, 0.0])),
            (limites["x_max"] - q[0], np.array([-1.0, 0.0])),
            (q[1] - limites["y_min"], np.array([0.0, 1.0])),
            (limites["y_max"] - q[1], np.array([0.0, -1.0])),
        ]
        fontes.extend((d, direcao) for d, direcao in paredes if d < d0)

        contorno = limites_mesa.contorno_de_limites(limites)
        if contorno is not None:
            p = limites_mesa.ponto_mais_proximo_contorno(q, contorno)
            dist = float(np.linalg.norm(q - p))
            fora = not limites_mesa.dentro_varios(q, contorno)[0]
            d = -dist if fora else dist
            if d < d0 and dist > 1e-9:
                fontes.append((d, (q - p) / dist))

    for d, direcao in fontes:
        if d <= 0.0:
            d_eff = 1e-3
            direcao = -direcao
        else:
            d_eff = d
        eta = 1.0 / d_eff - 1.0 / d0
        U += 0.5 * k_rep * eta ** 2 * d_objetivo ** 2
        grad += k_rep * (
            -eta * (1.0 / d_eff ** 2) * d_objetivo ** 2 * direcao
            + eta ** 2 * erro
        )

    return U, grad


def plan_potencial(inicio, objetivo, obstaculos, alpha=0.02, max_iter=5000,
                   tol=0.01, passo_max=0.01, max_escapes=20,
                   **kw_potencial):
    """Descida de gradiente sobre o potencial continuo."""
    limites = kw_potencial.get("limites")
    q = np.array(inicio, dtype=float)
    objetivo = np.array(objetivo, dtype=float)
    caminho = [q.copy()]

    def confinar(p):
        if not limites:
            return p
        p = np.array([
            np.clip(p[0], limites["x_min"], limites["x_max"]),
            np.clip(p[1], limites["y_min"], limites["y_max"]),
        ])
        contorno = limites_mesa.contorno_de_limites(limites)
        if contorno is not None and not limites_mesa.dentro_varios(p, contorno)[0]:
            p = limites_mesa.ponto_mais_proximo_contorno(p, contorno)
        return p

    def livre(p):
        return all(dist_ponto_poligono(p, o) > 0.0 for o in obstaculos)

    def sentido_contorno(p, direcao_grad):
        perp = np.array([-direcao_grad[1], direcao_grad[0]])
        return 1.0 if perp @ (objetivo - p) >= 0.0 else -1.0

    janela = 50
    passos_tangencial = 0
    sinal = 1.0
    tentativas = 0
    escapes = 0
    d_anterior = np.linalg.norm(q - objetivo)

    for it in range(max_iter):
        if np.linalg.norm(q - objetivo) < tol:
            return np.array(caminho), True

        _, grad = potencial(q, objetivo, obstaculos, **kw_potencial)
        norma = np.linalg.norm(grad)

        if it > 0 and it % janela == 0:
            d_atual = np.linalg.norm(q - objetivo)
            if d_atual > d_anterior - 1e-4 and passos_tangencial == 0:
                if escapes >= max_escapes:
                    break
                escapes += 1
                if tentativas == 0 and norma > 1e-9:
                    sinal = sentido_contorno(q, -grad / norma)
                else:
                    sinal = -sinal
                tentativas += 1
                passos_tangencial = (1 + tentativas) * janela
            elif d_atual < d_anterior - 1e-4:
                tentativas = 0
            d_anterior = d_atual

        if norma < 1e-9:
            if escapes >= max_escapes:
                break
            escapes += 1
            passos_tangencial = 2 * janela
            direcao = objetivo - q
            norma = np.linalg.norm(direcao)
            direcao = direcao / norma
            norma = passo_max / max(alpha, 1e-9)
            sinal = sentido_contorno(q, direcao) if tentativas == 0 else -sinal
            tentativas += 1
        else:
            direcao = -grad / norma

        if passos_tangencial > 0:
            direcao = sinal * np.array([-direcao[1], direcao[0]])
            passos_tangencial -= 1

        comprimento = min(alpha * norma, passo_max)
        perp = np.array([-direcao[1], direcao[0]])
        q_novo = None
        for escala in (1.0, 0.5, 0.25):
            for cand in (direcao, sinal * perp, -sinal * perp):
                tentativa = confinar(q + escala * comprimento * cand)
                if livre(tentativa):
                    q_novo = tentativa
                    break
            if q_novo is not None:
                break

        if q_novo is None:
            break

        q = q_novo
        caminho.append(q.copy())

    sucesso = np.linalg.norm(q - objetivo) < tol
    return np.array(caminho), sucesso


# ----------------------------------------- Grid + Brushfire + Potencial + A*

def construir_grid(obstaculos, limites, passo):
    """Rasteriza o workspace (no C-space) numa grade regular de ocupacao.

    Devolve (livre, origem): 'livre' e uma matriz booleana [altura x largura]
    com True nas celulas livres, e 'origem' e o canto (x_min, y_min).
    """
    x0, y0 = limites["x_min"], limites["y_min"]
    largura = max(1, int(np.ceil((limites["x_max"] - x0) / passo)))
    altura = max(1, int(np.ceil((limites["y_max"] - y0) / passo)))
    origem = np.array([x0, y0], dtype=float)

    ocupado = np.zeros((altura, largura), dtype=np.uint8)
    try:
        import cv2
    except ImportError:
        for i in range(altura):
            for j in range(largura):
                c = origem + (np.array([j, i]) + 0.5) * passo
                if any(ponto_dentro_poligono(c, o) for o in obstaculos):
                    ocupado[i, j] = 1
    else:
        for poligono in obstaculos:
            pts = np.round((np.asarray(poligono, dtype=float) - origem)
                           / passo - 0.5).astype(np.int32)
            cv2.fillPoly(ocupado, [pts], 1)

    contorno = limites_mesa.contorno_de_limites(limites)
    if contorno is not None:
        xs = origem[0] + (np.arange(largura) + 0.5) * passo
        ys = origem[1] + (np.arange(altura) + 0.5) * passo
        gx, gy = np.meshgrid(xs, ys)
        centros = np.column_stack([gx.ravel(), gy.ravel()])
        dentro = limites_mesa.dentro_varios(centros, contorno).reshape(altura, largura)
        ocupado[~dentro] = 1

    return ocupado == 0, origem


def mapa_brushfire(livre):
    """Algoritmo Brushfire: espalhamento a partir dos obstaculos e bordas.

    Calcula, por busca em largura (BFS 8-conectada), a distancia em passos
    de grade de cada celula livre ate a celula ocupada (obstaculo inflado
    ou parede da mesa) mais proxima:
      - 0 nas celulas ocupadas;
      - 1 nas celulas livres adjacentes a obstaculos ou a borda do grid;
      - 2, 3, ... conforme se afasta dos obstaculos.

    Esse mapa de distancias discretas alimenta o campo de potencial
    repulsivo usado pelo A* para manter o robo afastado dos obstaculos.
    """
    altura, largura = livre.shape
    dist = np.full(livre.shape, -1, dtype=int)
    fila = []

    for i in range(altura):
        for j in range(largura):
            if not livre[i, j]:
                dist[i, j] = 0
                fila.append((i, j))

    # As bordas externas do grid tambem contam como fronteira do workspace
    for i in range(altura):
        for j in range(largura):
            if livre[i, j] and (i == 0 or i == altura - 1 or j == 0 or j == largura - 1):
                dist[i, j] = 1
                fila.append((i, j))

    head = 0
    while head < len(fila):
        i, j = fila[head]
        head += 1
        d_atual = dist[i, j]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < altura and 0 <= nj < largura and dist[ni, nj] < 0:
                    dist[ni, nj] = d_atual + 1
                    fila.append((ni, nj))

    # Caso nao haja nenhum obstaculo/borda
    dist[dist < 0] = max(altura, largura)
    return dist


def potencial_repulsivo_brushfire(brushfire, passo, d0=0.18, peso_rep=6.0):
    """Converte o mapa Brushfire em um campo de potencial repulsivo no grid.

    Para cada celula com distancia d_m = brushfire * passo < d0:
        U_rep(c) = peso_rep * ((1/d_m - 1/d0) / (1/passo - 1/d0))^2
    Celulas com d_m >= d0 tem U_rep = 0. Celulas ocupadas recebem Inf.
    """
    pot = np.zeros(brushfire.shape, dtype=float)
    pot[brushfire <= 0] = np.inf
    if d0 <= passo or peso_rep <= 0.0:
        return pot

    d_m = brushfire.astype(float) * passo
    ativo = (brushfire > 0) & (d_m < d0)
    if np.any(ativo):
        escala_max = (1.0 / passo) - (1.0 / d0)
        eta = (1.0 / d_m[ativo]) - (1.0 / d0)
        pot[ativo] = peso_rep * ((eta / escala_max) ** 2)
    return pot


def mapa_wavefront(livre, objetivo_ij):
    """Wavefront (NF1) a partir do objetivo para visualizacao do campo na figura."""
    valores = np.full(livre.shape, -1, dtype=int)
    altura, largura = livre.shape
    if not livre[objetivo_ij]:
        return valores

    valores[objetivo_ij] = 0
    fila = [objetivo_ij]
    while fila:
        proxima = []
        for i, j in fila:
            v_atual = valores[i, j]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    a, b = i + di, j + dj
                    if not (0 <= a < altura and 0 <= b < largura):
                        continue
                    if not livre[a, b] or valores[a, b] >= 0:
                        continue
                    if di and dj and not (livre[a, j] and livre[i, b]):
                        continue
                    valores[a, b] = v_atual + 1
                    proxima.append((a, b))
        fila = proxima
    return valores


def _celula(ponto, origem, passo, forma):
    """Indice (i, j) da celula que contem 'ponto', limitado a grade."""
    j, i = np.floor((np.asarray(ponto, dtype=float) - origem) / passo)
    return (int(np.clip(i, 0, forma[0] - 1)),
            int(np.clip(j, 0, forma[1] - 1)))


def _centro(celula, origem, passo):
    """Coordenada (x, y) do centro da celula (i, j)."""
    i, j = celula
    return origem + (np.array([j, i], dtype=float) + 0.5) * passo


def _celula_livre_mais_proxima(celula, livre):
    """Celula livre mais proxima de 'celula', por busca em largura."""
    if livre[celula]:
        return celula
    altura, largura = livre.shape
    vistos = {celula}
    fila = [celula]
    while fila:
        proxima = []
        for i, j in fila:
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                v = (i + di, j + dj)
                if (0 <= v[0] < altura and 0 <= v[1] < largura
                        and v not in vistos):
                    if livre[v]:
                        return v
                    vistos.add(v)
                    proxima.append(v)
        fila = proxima
    return None


def busca_a_estrela(livre, inicio_ij, objetivo_ij, campo_repulsivo=None):
    """A* 8-conectado sobre o Grid guiado pelo Campo de Potencial (Brushfire).

    O custo de transicao para uma celula vizinha combina a distancia
    geometrica do passo (1 ou sqrt(2)) com o potencial repulsivo do
    Brushfire naquela celula:
        custo_passo = peso_geometrico * (1.0 + campo_repulsivo[v])
    E a heuristica h(c) e a distancia octil ate o objetivo (potencial
    atrativo). Assim o A* encontra o menor caminho que simultaneamente
    se mantem afastado dos obstaculos e das bordas da mesa, sem sofrer
    com minimos locais.
    """
    altura, largura = livre.shape
    raiz2 = np.sqrt(2.0)
    vizinhos = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, raiz2), (-1, 1, raiz2),
                (1, -1, raiz2), (1, 1, raiz2))

    def h(c):
        di = abs(c[0] - objetivo_ij[0])
        dj = abs(c[1] - objetivo_ij[1])
        return (raiz2 - 1.0) * min(di, dj) + max(di, dj)

    g = {inicio_ij: 0.0}
    veio_de = {}
    fechados = set()
    fronteira = [(h(inicio_ij), 0.0, inicio_ij)]

    while fronteira:
        _, custo, atual = heapq.heappop(fronteira)
        if atual in fechados:
            continue
        if atual == objetivo_ij:
            caminho = [atual]
            while atual in veio_de:
                atual = veio_de[atual]
                caminho.append(atual)
            caminho.reverse()
            return caminho
        fechados.add(atual)

        i, j = atual
        for di, dj, peso in vizinhos:
            v = (i + di, j + dj)
            if not (0 <= v[0] < altura and 0 <= v[1] < largura):
                continue
            if not livre[v] or v in fechados:
                continue
            if di and dj and not (livre[i + di, j] and livre[i, j + dj]):
                continue  # nao corta canto de celula ocupada
            rep = float(campo_repulsivo[v]) if campo_repulsivo is not None else 0.0
            novo = custo + peso * (1.0 + rep)
            if novo < g.get(v, np.inf):
                g[v] = novo
                veio_de[v] = atual
                heapq.heappush(fronteira, (novo + h(v), novo, v))
    return None


def _visada_livre(a, b, livre, origem, passo, brushfire=None, min_bf=1):
    """Verdadeiro se o segmento ab atravessa apenas celulas com brushfire >= min_bf."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = int(np.ceil(np.linalg.norm(b - a) / (0.5 * passo))) + 1
    for t in np.linspace(0.0, 1.0, n):
        c = _celula(a + t * (b - a), origem, passo, livre.shape)
        if not livre[c]:
            return False
        if brushfire is not None and brushfire[c] < min_bf:
            return False
    return True


def suavizar_caminho(pontos, livre, origem, passo, brushfire=None, bf_seguro=4):
    """Remove os degraus da grade por atalhos em linha de visada segura.

    Quando 'brushfire' e fornecido, um atalho entre pontos[i] e pontos[j]
    so e aceito se nao chegar mais perto dos obstaculos do que o proprio
    trecho original (limitado a 'bf_seguro' celulas de folga). Isso evita
    que a suavizacao cole a trajetoria de volta na quina dos obstaculos.
    """
    if len(pontos) <= 2:
        return np.asarray(pontos, dtype=float)

    saida = [pontos[0]]
    i = 0
    while i < len(pontos) - 1:
        j = len(pontos) - 1
        while j > i + 1:
            min_bf = 1
            if brushfire is not None:
                bfs_trecho = [
                    int(brushfire[_celula(pontos[k], origem, passo, livre.shape)])
                    for k in range(i, j + 1)
                ]
                min_bf = max(1, min(min(bfs_trecho), bf_seguro))
            if _visada_livre(pontos[i], pontos[j], livre, origem, passo,
                             brushfire=brushfire, min_bf=min_bf):
                break
            j -= 1
        saida.append(pontos[j])
        i = j
    return np.asarray(saida, dtype=float)


def _dentro_poligono_varios(pontos, poligono):
    """Ray casting vetorizado: mascara dos pontos dentro do poligono."""
    x, y = pontos[:, 0], pontos[:, 1]
    dentro = np.zeros(len(pontos), dtype=bool)
    n = len(poligono)
    for i in range(n):
        x1, y1 = poligono[i]
        x2, y2 = poligono[(i + 1) % n]
        if y1 == y2:
            continue
        cruza = (y1 > y) != (y2 > y)
        x_cruz = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        dentro ^= cruza & (x < x_cruz)
    return dentro


def _visada_poligonos(a, b, obstaculos, limites=None, passo=0.005):
    """Testa linha de visada diretamente contra os poligonos."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = int(np.ceil(np.linalg.norm(b - a) / passo)) + 1
    amostras = a + np.linspace(0.0, 1.0, n).reshape(-1, 1) * (b - a)

    if limites:
        if (np.any(amostras[:, 0] < limites["x_min"])
                or np.any(amostras[:, 0] > limites["x_max"])
                or np.any(amostras[:, 1] < limites["y_min"])
                or np.any(amostras[:, 1] > limites["y_max"])):
            return False
        if not limites_mesa.amostras_dentro(amostras, limites).all():
            return False

    for poligono in obstaculos:
        if _dentro_poligono_varios(amostras,
                                   np.asarray(poligono, dtype=float)).any():
            return False
    return True


def limitar_pontos(pontos, maximo, visada_livre):
    """Reduz a poligonal a no maximo 'maximo' pontos, sem colidir."""
    pontos = [np.asarray(p, dtype=float) for p in pontos]
    if not maximo or maximo <= 0 or len(pontos) <= maximo:
        return np.asarray(pontos, dtype=float), True
    if maximo < 2:
        return np.asarray([pontos[0], pontos[-1]], dtype=float), False

    n = len(pontos)
    anterior = list(range(-1, n - 1))
    seguinte = list(range(1, n + 1))
    seguinte[-1] = -1
    vivo = [True] * n
    versao = [0] * n
    restantes = n

    def desvio(k):
        a, b, c = pontos[anterior[k]], pontos[k], pontos[seguinte[k]]
        return float(np.linalg.norm(b - ponto_mais_proximo_segmento(b, a, c)))

    fila = [(desvio(k), 0, k) for k in range(1, n - 1)]
    heapq.heapify(fila)

    while restantes > maximo and fila:
        _, v, k = heapq.heappop(fila)
        if not vivo[k] or v != versao[k]:
            continue
        a, c = anterior[k], seguinte[k]
        if not visada_livre(pontos[a], pontos[c]):
            continue
        vivo[k] = False
        restantes -= 1
        seguinte[a] = c
        anterior[c] = a
        for m in (a, c):
            if 0 < m < n - 1 and vivo[m]:
                versao[m] += 1
                heapq.heappush(fila, (desvio(m), versao[m], m))

    saida = np.asarray([p for p, ok in zip(pontos, vivo) if ok], dtype=float)
    return saida, len(saida) <= maximo


def reamostrar_n(pontos, n):
    """Reamostra a poligonal em exatamente n pontos mantendo os vertices."""
    pontos = np.asarray(pontos, dtype=float)
    if len(pontos) < 2 or n <= len(pontos):
        return pontos

    distancias = np.linalg.norm(np.diff(pontos, axis=0), axis=1)
    total = float(distancias.sum())
    if total <= 0.0:
        return pontos

    extras = n - len(pontos)
    quota = extras * distancias / total
    por_segmento = np.floor(quota).astype(int)
    sobra = extras - int(por_segmento.sum())
    if sobra > 0:
        ordem = np.argsort(-(quota - por_segmento))
        por_segmento[ordem[:sobra]] += 1

    saida = [pontos[0]]
    for a, b, k in zip(pontos[:-1], pontos[1:], por_segmento):
        for m in range(1, k + 1):
            saida.append(a + (b - a) * (m / (k + 1)))
        saida.append(b)
    return np.asarray(saida)


def reduzir_pontos(pontos, livre, origem, passo,
                   minimo=MIN_PONTOS_TRAJETORIA,
                   maximo=MAX_PONTOS_TRAJETORIA,
                   brushfire=None, bf_seguro=3):
    """Garante que a trajetoria respeite o limite maximo de pontos (16 = inicio + 15 WPs)."""
    pontos = np.asarray(pontos, dtype=float)

    if maximo and len(pontos) > maximo:
        antes = len(pontos)

        def visada_com_folga(a, b):
            if brushfire is None:
                return _visada_livre(a, b, livre, origem, passo)
            ca = _celula(a, origem, passo, livre.shape)
            cb = _celula(b, origem, passo, livre.shape)
            folga = max(1, min(int(brushfire[ca]), int(brushfire[cb]), bf_seguro))
            return _visada_livre(a, b, livre, origem, passo,
                                 brushfire=brushfire, min_bf=folga)

        pontos, cabe = limitar_pontos(pontos, maximo, visada_com_folga)
        if not cabe and brushfire is not None:
            # Se nao coube com a folga do Brushfire, relaxa para visada livre no C-space
            pontos, cabe = limitar_pontos(
                pontos, maximo,
                lambda a, b: _visada_livre(a, b, livre, origem, passo))
        print(f"[OK] Trajetoria reduzida de {antes} para {len(pontos)} "
              f"pontos ({len(pontos) - 1} waypoints, limite {maximo - 1} waypoints)")
        if not cabe:
            print(f"[AVISO] Nao da para descer de {len(pontos)} pontos "
                  f"sem cortar obstaculo")

    if minimo and len(pontos) < minimo:
        pontos = reamostrar_n(pontos, minimo)
    return pontos


def plan_grid(inicio, objetivo, obstaculos, limites, passo=0.02,
              suavizar=True, min_pontos=MIN_PONTOS_TRAJETORIA,
              max_pontos=MAX_PONTOS_TRAJETORIA,
              d0_rep=0.18, peso_rep=6.0):
    """Planejamento em Grid com Campo de Potencial (Brushfire) e busca A*.

    1. Constroi o grid de ocupacao sobre os obstaculos expandidos (Minkowski).
    2. Calcula o mapa Brushfire (espalhamento da distancia aos obstaculos).
    3. Gera o campo de potencial repulsivo a partir do Brushfire.
    4. Executa o A* ponderado pelo potencial para achar o menor caminho seguro.
    5. Suaviza e limita o caminho a no maximo 15 waypoints (16 pontos).
    """
    livre, origem = construir_grid(obstaculos, limites, passo)
    brushfire = mapa_brushfire(livre)
    campo_rep = potencial_repulsivo_brushfire(brushfire, passo, d0=d0_rep, peso_rep=peso_rep)

    inicio_ij = _celula(inicio, origem, passo, livre.shape)
    objetivo_ij = _celula(objetivo, origem, passo, livre.shape)

    for nome, c in (("inicio", inicio_ij), ("objetivo", objetivo_ij)):
        if not livre[c]:
            novo = _celula_livre_mais_proxima(c, livre)
            if novo is None or max(abs(novo[0] - c[0]),
                                   abs(novo[1] - c[1])) > 2:
                print(f"[ERRO] O {nome} esta em regiao proibida do espaco "
                      f"de configuracoes (obstaculo inflado pelo raio do "
                      f"robo): escolha outro ponto ou reduza --raio-robo")
                return np.array([np.asarray(inicio, dtype=float)]), False
            print(f"[AVISO] Celula do {nome} ocupada na grade; "
                  f"usando a celula livre vizinha")
            if nome == "inicio":
                inicio_ij = novo
            else:
                objetivo_ij = novo

    celulas = busca_a_estrela(livre, inicio_ij, objetivo_ij, campo_repulsivo=campo_rep)
    n_livres = int(livre.sum())
    print(f"[OK] Grade {livre.shape[1]}x{livre.shape[0]} celulas de "
          f"{passo} m ({n_livres} livres, "
          f"{livre.size - n_livres} ocupadas, "
          f"Brushfire max={brushfire.max()} celulas)")
    if celulas is None:
        print("[AVISO] Nao existe caminho na resolucao pedida "
              "(tente --passo menor ou --margem menor)")
        return np.array([np.asarray(inicio, dtype=float)]), False

    pontos = [np.asarray(inicio, dtype=float)]
    pontos += [_centro(c, origem, passo) for c in celulas]
    pontos.append(np.asarray(objetivo, dtype=float))

    bf_seguro = max(2, int(round(0.5 * d0_rep / passo))) if d0_rep > 0 else 1
    if suavizar:
        pontos = suavizar_caminho(pontos, livre, origem, passo,
                                  brushfire=brushfire, bf_seguro=bf_seguro)
    pontos = np.asarray(pontos, dtype=float)

    if max_pontos or min_pontos:
        pontos = reduzir_pontos(pontos, livre, origem, passo,
                                minimo=min_pontos, maximo=max_pontos,
                                brushfire=brushfire, bf_seguro=bf_seguro)
    return pontos, True


# ------------------------------------------------------------------- mapa

def carregar_mapa(caminho_json):
    """Le obstaculos, limites e pose do robo do JSON de 'gerar_mapa.py'."""
    with open(caminho_json, "r", encoding="utf-8") as f:
        dados = json.load(f)

    obstaculos = [
        np.array(obs["vertices_xy_m"], dtype=float)
        for obs in dados.get("obstaculos", [])
    ]
    limites = dados.get("workspace_limites_m", {})

    contorno = dados.get("workspace_contorno_xy_m")
    if contorno and limites:
        limites["contorno"] = np.array(contorno, dtype=float)

    return obstaculos, limites, dados.get("robo")


def pose_inicial(robo):
    """Ponto inicial (x, y) do bloco 'robo', ou None se faltar."""
    if not robo or "x_m" not in robo or "y_m" not in robo:
        return None
    return [float(robo["x_m"]), float(robo["y_m"])]


def mapa_exemplo():
    """Mapa sintetico usado quando nao ha JSON disponivel."""
    return [
        poligono_regular((0.3, 0.3), 4, 0.08),
        poligono_regular((0.7, 0.5), 3, 0.10),
        poligono_regular((0.5, 0.65), 6, 0.09),
    ], {"x_min": 0.0, "x_max": 1.2, "y_min": 0.0, "y_max": 0.8}, None


def plotar_valores(caminho, obstaculos, inicio, objetivo, grid, arquivo):
    """Desenha a grade como tabela com o valor em cada celula."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    livre, origem, passo, valores = grid
    altura, largura = valores.shape

    def celula_de(p):
        return _celula(p, origem, passo, valores.shape)

    celulas_caminho = {celula_de(p) for p in caminho}
    celula_inicio = celula_de(inicio)
    celula_objetivo = celula_de(objetivo)

    fig, ax = plt.subplots(figsize=(0.42 * largura + 2,
                                    0.42 * altura + 1.5))

    for i in range(altura):
        for j in range(largura):
            x = origem[0] + j * passo
            y = origem[1] + i * passo
            v = valores[i, j]

            if v < 0:
                cor_fundo, cor_texto, texto = "0.78", "0.25", "Inf"
            else:
                cor_fundo, cor_texto, texto = "white", "0.15", str(v)
            if (i, j) in celulas_caminho:
                cor_fundo = "#bcd9f2"
            if (i, j) == celula_objetivo:
                cor_fundo, cor_texto = "#f4a0a0", "k"
            elif (i, j) == celula_inicio:
                cor_fundo, cor_texto = "#a8dfa8", "k"

            ax.add_patch(Rectangle((x, y), passo, passo,
                                   facecolor=cor_fundo, edgecolor="0.55",
                                   lw=0.6, zorder=0))
            ax.text(x + passo / 2, y + passo / 2, texto, ha="center",
                    va="center", fontsize=6.5, color=cor_texto, zorder=1)

    for poligono in obstaculos:
        ax.add_patch(Polygon(poligono, closed=True, fill=False,
                             edgecolor="k", lw=1.6, zorder=2))

    ax.plot(caminho[:, 0], caminho[:, 1], "-b", marker="o", ms=5, lw=2.0, zorder=3)
    ax.plot(*inicio, "go", ms=9, mec="k", zorder=4)
    ax.plot(*objetivo, "r*", ms=15, mec="k", zorder=4)

    ax.set_xlim(origem[0], origem[0] + largura * passo)
    ax.set_ylim(origem[1], origem[1] + altura * passo)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"valores da grade (brushfire/wavefront) - celula de {passo} m",
                 fontsize=10)
    fig.savefig(arquivo, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figura salva em '{arquivo}'")


def plotar(caminho, obstaculos, expandidos, inicio, objetivo, limites,
           arquivo, grid=None):
    """Desenha mapa, expansao de Minkowski, campo potencial do grid e trajetoria."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(8, 6))

    if grid is not None:
        livre, origem, passo, valores = grid
        altura, largura = livre.shape
        xs = origem[0] + (np.arange(largura) + 0.5) * passo
        ys = origem[1] + (np.arange(altura) + 0.5) * passo
        gx, gy = np.meshgrid(xs, ys)
        pontos_grid = [
            ax.plot(gx[livre], gy[livre], ".", color="#5b9bd5",
                    ls="none", zorder=0, label="celula livre")[0],
            ax.plot(gx[~livre], gy[~livre], ".", color="#d6604d",
                    ls="none", zorder=0, label="celula ocupada")[0],
        ]

        if valores is not None:
            campo = np.ma.masked_less(valores, 0)
            imagem = ax.imshow(
                campo, origin="lower", zorder=-1, cmap="viridis", alpha=0.75,
                extent=(origem[0], origem[0] + largura * passo,
                        origem[1], origem[1] + altura * passo))
            barra = fig.colorbar(imagem, ax=ax, fraction=0.046, pad=0.04)
            barra.set_label("campo no grid (passos ate o objetivo / brushfire)",
                            fontsize=8)
            barra.ax.tick_params(labelsize=7)

            if campo.count() <= 400:
                for i in range(altura):
                    for j in range(largura):
                        if valores[i, j] >= 0:
                            ax.text(xs[j], ys[i], str(valores[i, j]),
                                    ha="center", va="center", fontsize=5,
                                    color="w", zorder=1)

    mostrar_margem = len(expandidos) != len(obstaculos) or any(
        len(a) != len(b) for a, b in zip(expandidos, obstaculos))
    if mostrar_margem:
        for i, poligono in enumerate(expandidos):
            ax.add_patch(Polygon(
                poligono, closed=True, facecolor="0.85", edgecolor="r",
                ls="--", lw=1.2,
                label="expansao Minkowski" if i == 0 else None))
    for i, poligono in enumerate(obstaculos):
        ax.add_patch(Polygon(poligono, closed=True, facecolor="0.6",
                             edgecolor="k",
                             alpha=0.45 if grid is not None else 1.0,
                             label="obstaculo" if i == 0 else None))
    ax.plot(caminho[:, 0], caminho[:, 1], "-bo", ms=5, lw=2,
            label=f"trajetoria ({len(caminho) - 1} waypoints)")
    ax.plot(*inicio, "go", ms=10, label="inicio")
    ax.plot(*objetivo, "r*", ms=15, label="objetivo")
    contorno_mesa = limites_mesa.contorno_de_limites(limites)
    if contorno_mesa is not None:
        ax.add_patch(Polygon(contorno_mesa, closed=True, fill=False,
                             edgecolor="m", lw=2, label="limite da mesa"))
    ax.set_xlim(limites.get("x_min", 0.0), limites.get("x_max", 1.2))
    ax.set_ylim(limites.get("y_min", 0.0), limites.get("y_max", 0.8))
    if limites.get("origem") == "superior_esquerdo":
        ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if grid is not None:
        fig.canvas.draw()
        caixa = ax.get_window_extent()
        dy_lim = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
        pts_por_m = (caixa.height / max(dy_lim, 1e-6)) * 72.0 / fig.dpi
        for artista in pontos_grid:
            artista.set_markersize(max(0.3, 0.45 * pts_por_m * passo))

    legenda = ax.legend(loc="best", fontsize=8)
    if grid is not None:
        handles = getattr(legenda, "legend_handles",
                          getattr(legenda, "legendHandles", []))
        for handle in handles[:len(pontos_grid)]:
            handle.set_markersize(7)
    ax.grid(True, alpha=0.3)
    fig.savefig(arquivo, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figura salva em '{arquivo}'")


def plotar_opencv(caminho, obstaculos, expandidos, inicio, objetivo, limites,
                  arquivo, grid=None):
    """Fallback para gerar a figura usando OpenCV quando matplotlib nao esta instalado."""
    import cv2

    x_min = float(limites.get("x_min", 0.0))
    x_max = float(limites.get("x_max", 2.5))
    y_min = float(limites.get("y_min", 0.0))
    y_max = float(limites.get("y_max", 1.21))
    ppm = 420.0
    largura_px = max(200, int(round((x_max - x_min) * ppm)))
    altura_px = max(200, int(round((y_max - y_min) * ppm)))
    sup_esq = limites.get("origem") == "superior_esquerdo"

    img = np.full((altura_px, largura_px, 3), 245, dtype=np.uint8)

    def para_px(pts):
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        px = np.round((pts[:, 0] - x_min) * ppm).astype(np.int32)
        if sup_esq:
            py = np.round((pts[:, 1] - y_min) * ppm).astype(np.int32)
        else:
            py = np.round((y_max - pts[:, 1]) * ppm).astype(np.int32)
        return np.column_stack([px, py])

    if grid is not None:
        livre, origem, passo, valores = grid
        gh, gw = livre.shape
        for i in range(gh):
            for j in range(gw):
                c_xy = origem + (np.array([j, i]) + 0.5) * passo
                p = para_px(c_xy)[0]
                cor = (225, 195, 150) if livre[i, j] else (140, 140, 220)
                cv2.circle(img, (int(p[0]), int(p[1])), 2, cor, -1)

    for pol in expandidos:
        pts_px = para_px(pol)
        cv2.fillPoly(img, [pts_px], (220, 220, 235))
        cv2.polylines(img, [pts_px], True, (0, 0, 220), 2)

    for pol in obstaculos:
        pts_px = para_px(pol)
        cv2.fillPoly(img, [pts_px], (145, 145, 145))
        cv2.polylines(img, [pts_px], True, (20, 20, 20), 2)

    contorno_mesa = limites_mesa.contorno_de_limites(limites)
    if contorno_mesa is not None:
        cv2.polylines(img, [para_px(contorno_mesa)], True, (200, 0, 200), 2)

    if len(caminho) >= 2:
        cam_px = para_px(caminho)
        cv2.polylines(img, [cam_px], False, (220, 60, 0), 3)
        for p in cam_px[1:-1]:
            cv2.circle(img, (int(p[0]), int(p[1])), 5, (180, 0, 0), -1)

    ini_px = para_px(inicio)[0]
    obj_px = para_px(objetivo)[0]
    cv2.circle(img, (int(ini_px[0]), int(ini_px[1])), 8, (0, 180, 0), -1)
    cv2.circle(img, (int(obj_px[0]), int(obj_px[1])), 8, (0, 0, 230), -1)

    cv2.imwrite(arquivo, img)
    print(f"[OK] Figura salva em '{arquivo}' (via OpenCV)")


# ------------------------------------------- Comunicacao com o Robo (demo.py)

def list_ports():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("Nenhuma porta serial encontrada.")
        return
    print("Portas disponíveis:")
    for p in ports:
        print(f"  {p.device:20s}  {p.description}")


def connect(port=DEFAULT_PORT, baud=DEFAULT_BAUD):
    ser = serial.Serial(port, baud, timeout=TIMEOUT_S)
    time.sleep(1.5)
    ser.reset_input_buffer()
    return ser


def send(ser, cmd):
    ser.write(f"<{cmd}>\n".encode("ascii"))
    ser.flush()


def read_response(ser, timeout=RESPONSE_TIMEOUT_S):
    t0 = time.time()
    buf = bytearray()
    while time.time() - t0 < timeout:
        b = ser.read(1)
        if not b:
            continue
        buf += b
        if b == b"\n":
            line = buf.decode("ascii", errors="replace").strip()
            if line.startswith("<") and line.endswith(">"):
                return line
            buf = bytearray()
    return None


def cmd_and_wait(ser, cmd, timeout=RESPONSE_TIMEOUT_S):
    send(ser, cmd)
    return read_response(ser, timeout)


def add_waypoint(ser, x, y, th=0.0):
    return cmd_and_wait(ser, f"A,{x:.3f},{y:.3f},{th:.1f}")


def set_waypoints(ser, pts):
    if not pts:
        return cmd_and_wait(ser, "C")
    body = f"W,{len(pts)}"
    for pt in pts:
        x, y = float(pt[0]), float(pt[1])
        th = float(pt[2]) if len(pt) >= 3 else 0.0
        body += f",{x:.3f},{y:.3f},{th:.1f}"
    return cmd_and_wait(ser, body)


def append_waypoints(ser, pts):
    if not pts:
        return None
    body = f"Q,{len(pts)}"
    for pt in pts:
        x, y = float(pt[0]), float(pt[1])
        th = float(pt[2]) if len(pt) >= 3 else 0.0
        body += f",{x:.3f},{y:.3f},{th:.1f}"
    return cmd_and_wait(ser, body)


def list_waypoints(ser):
    resp = cmd_and_wait(ser, "L")
    if resp is None or not resp.startswith("<L,"):
        return None
    parts = resp[1:-1].split(",")[1:]
    n = int(parts[0])
    wps = []
    for i in range(n):
        x = float(parts[1 + 3 * i])
        y = float(parts[2 + 3 * i])
        th = float(parts[3 + 3 * i])
        wps.append((x, y, th))
    return wps


def delete_waypoint(ser, idx):
    return cmd_and_wait(ser, f"D,{idx}")


def replace_waypoint(ser, idx, x, y, th=0.0):
    return cmd_and_wait(ser, f"S,{idx},{x:.3f},{y:.3f},{th:.1f}")


def start(ser, wait_auto_eq=True):
    resp = cmd_and_wait(ser, "R")
    if resp is None:
        return False, None, "sem resposta (timeout)"
    if resp != "<OK>":
        if resp == "<ERR>":
            return False, resp, "lista vazia — envie waypoints antes de <R>"
        return False, resp, f"resposta inesperada: {resp}"
    if wait_auto_eq:
        time.sleep(START_GRACE_S)
    return True, resp, "navegação iniciada"


def stop(ser):
    return cmd_and_wait(ser, "C")


def goto(ser, x, y, th=0.0):
    return cmd_and_wait(ser, f"G,{x:.3f},{y:.3f},{th:.1f}")


def set_pose(ser, x, y, th):
    return cmd_and_wait(ser, f"O,{x:.3f},{y:.3f},{th:.1f}")


def get_telemetry(ser, timeout=RESPONSE_TIMEOUT_S):
    """Envia <T> e devolve dict com pose/estado do firmware, ou None."""
    resp = cmd_and_wait(ser, "T", timeout)
    if resp is None or not resp.startswith("<T,"):
        return None
    parts = resp[3:-1].split(",")
    if len(parts) != 7:
        return None
    try:
        return {
            "x": float(parts[0]),
            "y": float(parts[1]),
            "th_deg": float(parts[2]),
            "wp": int(parts[3]),
            "state": int(parts[4]),
            "v": float(parts[5]),
            "w_deg": float(parts[6]),
        }
    except ValueError:
        return None


def format_telemetry(t):
    if t is None:
        return "(sem telemetria)"
    st = STATE_NAMES.get(t["state"], "?")
    return (
        f"x={t['x']:+.3f}m  y={t['y']:+.3f}m  "
        f"th={t['th_deg']:+7.2f}°  "
        f"wp={t['wp']:2d}  st={st:5s}  "
        f"v={t['v']:+.3f}m/s  w={t['w_deg']:+6.2f}°/s"
    )


def wait_until_done_telemetry(ser, poll_s=0.4, timeout_s=120.0):
    """Monitora via <T> até pos_state == DONE (2) ou timeout."""
    t0 = time.time()
    last_wp = -1
    while time.time() - t0 < timeout_s:
        t = get_telemetry(ser)
        if t is None:
            time.sleep(poll_s)
            continue
        if t["wp"] != last_wp:
            print(f"  [{time.time()-t0:5.1f}s] WP {t['wp']} -> {format_telemetry(t)}")
            last_wp = t["wp"]
        else:
            print(f"  [{time.time()-t0:5.1f}s] {format_telemetry(t)}")
        if t["state"] == 2:  # DONE
            return True, t
        time.sleep(poll_s)
    return False, None


def compensar_deriva_odometria(pos_ini, waypoints, fator_escala=1.0,
                               comp_angular_graus=0.0):
    """Pre-compensa erros sistematicos da odometria interna do robo sem camera.

    - 'fator_escala' (< 1.0 quando o robo passa do objetivo): encurta cada
      segmento enviado ao firmware para que o deslocamento fisico real bata
      com a distancia planejada no mapa.
    - 'comp_angular_graus': gira os vetores de deslocamento em torno de
      pos_ini para compensar eventual tendencia sistematica do robo de
      puxar para a esquerda ou direita durante a caminhada.
    """
    wps = np.asarray(waypoints, dtype=float).copy()
    if len(wps) == 0:
        return wps

    vetores = wps - pos_ini
    if abs(fator_escala - 1.0) > 1e-6:
        vetores = vetores * float(fator_escala)

    if abs(comp_angular_graus) > 1e-6:
        ang = np.radians(float(comp_angular_graus))
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s], [s, c]], dtype=float)
        vetores = vetores @ R.T

    return pos_ini + vetores


def main():
    ap = argparse.ArgumentParser(
        description="Gera trajetoria por Campos de Potencial em Grid "
                    "(Brushfire + A* + Minkowski) e envia ao robo.")
    ap.add_argument("--mapa", default="mapa_obstaculos.json")
    ap.add_argument("--inicio", type=float, nargs=2, default=None,
                    help="ponto inicial [m]; o padrao e a pose do robo "
                         "lida do mapa")
    ap.add_argument("--objetivo", type=float, nargs=2, default=[0.7, 0.6],
                    help="ponto objetivo [x y] em metros (padrao: 0.7 0.6)")
    ap.add_argument("--metodo", choices=("grid", "potencial"), default="grid",
                    help="grid: Brushfire + Potencial + A* sobre grade (padrao); "
                         "potencial: descida de gradiente continua")
    ap.add_argument("--passo", type=float, default=0.02,
                    help="[grid] lado da celula da grade [m] (padrao 0.02)")
    ap.add_argument("--sem-suavizar", action="store_true",
                    help="[grid] mantem o caminho em degraus da grade")
    ap.add_argument("--min-pontos", type=int,
                    default=MIN_PONTOS_TRAJETORIA,
                    help=f"minimo de pontos da trajetoria (padrao "
                         f"{MIN_PONTOS_TRAJETORIA}); 0 nao subdivide retas")
    ap.add_argument("--max-pontos", type=int,
                    default=MAX_PONTOS_TRAJETORIA,
                    help=f"maximo de pontos contando o inicio (padrao "
                         f"{MAX_PONTOS_TRAJETORIA} = inicio + {MAX_WAYPOINTS_ROBO} waypoints)")
    ap.add_argument("--d0", type=float, default=0.15,
                    help="raio de influencia do campo repulsivo (Brushfire) [m] (padrao 0.18)")
    ap.add_argument("--peso-rep", type=float, default=5.0,
                    help="[grid] ganho do potencial repulsivo do Brushfire no A* (padrao 6.0)")
    ap.add_argument("--k-attr", type=float, default=1.0)
    ap.add_argument("--k-rep", type=float, default=0.001)
    ap.add_argument("--raio-robo", type=float, default=0.07,
                    help="raio do robo [m] para a Soma de Minkowski (padrao 0.14)")
    ap.add_argument("--margem", type=float, default=0.04,
                    help="folga extra de seguranca [m] somada ao raio do robo (padrao 0.02)")
    ap.add_argument("--fator-odometria", type=float, default=0.88,
                    help="fator de escala aplicado aos deslocamentos para evitar que o robo "
                         "passe do objetivo (padrao 0.88; use 1.0 para desativar)")
    ap.add_argument("--comp-angular-graus", type=float, default=0.0,
                    help="compensacao angular (graus) aplicada aos waypoints para corrigir "
                         "tendencia lateral da odometria (padrao 0.0)")
    ap.add_argument("--inverter-y-robo", action="store_true",
                    help="inverte o sinal de dy e theta enviados ao robo caso o firmware use "
                         "eixo Y oposto ao do mapa")
    ap.add_argument("--sem-telemetria", action="store_true",
                    help="nao aguarda/monitora a telemetria <T> apos iniciar a navegacao")
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--saida", default="trajetoria.json")
    ap.add_argument("--figura", default="trajetoria.png")
    ap.add_argument("--sem-envio", action="store_true",
                    help="so gera a trajetoria e a figura, sem abrir a "
                         "porta serial nem enviar waypoints")
    ap.add_argument("--port", default=DEFAULT_PORT,
                    help="Porta serial utilizada para a comunicacao com o robo")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD,
                    help="Baudrate utilizado para a comunicacao com o robo")

    args = ap.parse_args()

    # Garante a restricao do professor: no maximo 15 waypoints alem do ponto inicial
    if args.max_pontos <= 0 or args.max_pontos > MAX_PONTOS_TRAJETORIA:
        print(f"[INFO] Ajustando --max-pontos para {MAX_PONTOS_TRAJETORIA} "
              f"(1 ponto inicial + {MAX_WAYPOINTS_ROBO} waypoints maximos).")
        args.max_pontos = MAX_PONTOS_TRAJETORIA

    ser = None
    try:
        if not args.sem_envio:
            ser = connect(args.port, args.baud)
    except serial.SerialException as e:
        print(f"[AVISO] Não consegui abrir {args.port} @ {args.baud}: {e}")
        print("Verifique: HC-05 emparelhado, Monitor Serial fechado, porta correta.")
        sys.exit(1)

    try:
        obstaculos, limites, robo = carregar_mapa(args.mapa)
        print(f"[OK] {len(obstaculos)} obstaculos lidos de '{args.mapa}'")
    except FileNotFoundError:
        obstaculos, limites, robo = mapa_exemplo()
        print(f"[AVISO] '{args.mapa}' nao encontrado; usando mapa de exemplo")

    if args.inicio is None:
        args.inicio = pose_inicial(robo)
        if args.inicio is None:
            print("[ERRO] O mapa nao traz a pose do robo (bloco 'robo'); "
                  "informe --inicio x y")
            return
        theta = robo.get("theta_graus")
        print(f"[OK] Inicio lido do mapa: {args.inicio} "
              + (f"(theta {theta} graus)" if theta is not None else ""))

    # Soma de Minkowski: obstaculos inflados pelo raio do robo + margem
    raio = args.raio_robo + args.margem
    expandidos = expandir_obstaculos(obstaculos, raio, limites)
    limites_c = encolher_limites(limites, raio)
    if raio > 0:
        print(f"[OK] Obstaculos inflados por {raio:.3f} m "
              f"(robo {args.raio_robo} m + folga {args.margem} m): "
              f"{len(obstaculos)} -> {len(expandidos)} poligonos")

    for nome, ponto in (("inicio", args.inicio), ("objetivo", args.objetivo)):
        if not limites_mesa.ponto_dentro(ponto, limites_c):
            print(f"[AVISO] O {nome} {ponto} esta proximo da borda da mesa "
                  f"(dentro da margem de {raio:.2f} m); ajustando para celula livre mais proxima.")
        if any(dist_ponto_poligono(np.array(ponto), o) <= 0
               for o in expandidos):
            print(f"[AVISO] O {nome} {ponto} toca na margem inflada de um obstaculo; "
                  f"o planejador buscara a celula livre vizinha.")

    if args.max_pontos < 2:
        print("[ERRO] --max-pontos precisa ser >= 2 (inicio e objetivo)")
        return
    if args.min_pontos > args.max_pontos:
        print(f"[ERRO] --min-pontos ({args.min_pontos}) nao pode passar de "
              f"--max-pontos ({args.max_pontos})")
        return

    if args.metodo == "grid":
        caminho, sucesso = plan_grid(
            args.inicio, args.objetivo, expandidos, limites_c,
            passo=args.passo, suavizar=not args.sem_suavizar,
            min_pontos=args.min_pontos, max_pontos=args.max_pontos,
            d0_rep=args.d0, peso_rep=args.peso_rep)
    else:
        caminho, sucesso = plan_potencial(
            args.inicio, args.objetivo, expandidos,
            alpha=args.alpha, max_iter=args.max_iter,
            k_attr=args.k_attr, k_rep=args.k_rep, d0=args.d0,
            limites=limites_c)
        if args.max_pontos and len(caminho) > args.max_pontos:
            antes = len(caminho)
            caminho, cabe = limitar_pontos(
                caminho, args.max_pontos,
                lambda a, b: _visada_poligonos(a, b, expandidos, limites_c))
            print(f"[OK] Trajetoria reduzida de {antes} para "
                  f"{len(caminho)} pontos (limite {args.max_pontos})")
            if not cabe:
                print(f"[AVISO] Nao foi possivel descer a "
                      f"{args.max_pontos} pontos sem cortar obstaculo; "
                      f"ficaram {len(caminho)}")

    if sucesso:
        comprimento = float(np.sum(np.linalg.norm(
            np.diff(caminho, axis=0), axis=1)))
        print(f"[OK] Objetivo alcancado: {len(caminho) - 1} waypoints "
              f"({len(caminho)} pontos contando inicio), "
              f"comprimento {comprimento:.4f} m")
    elif args.metodo == "grid":
        print("[AVISO] Nenhum caminho encontrado")
    else:
        print(f"[AVISO] Parou em {caminho[-1].round(4).tolist()} "
              f"apos {len(caminho)} passos (possivel minimo local)")

    theta_inicial = robo.get("theta_rad") if robo else None
    theta_inicial_graus = robo.get("theta_graus") if robo else None

    with open(args.saida, "w", encoding="utf-8") as f:
        json.dump({
            "inicio_xy_m": list(args.inicio),
            "objetivo_xy_m": list(args.objetivo),
            "theta_inicial_robo_rad": theta_inicial,
            "theta_inicial_robo_graus": theta_inicial_graus,
            "metodo": args.metodo,
            "passo_grid_m": args.passo if args.metodo == "grid" else None,
            "d0_brushfire_m": args.d0 if args.metodo == "grid" else None,
            "peso_rep_brushfire": args.peso_rep if args.metodo == "grid" else None,
            "sucesso": bool(sucesso),
            "num_pontos": int(len(caminho)),
            "num_waypoints": int(max(0, len(caminho) - 1)),
            "max_waypoints": MAX_WAYPOINTS_ROBO,
            "raio_robo_m": args.raio_robo,
            "margem_seguranca_m": args.margem,
            "raio_inflado_m": raio,
            "fator_odometria": args.fator_odometria,
            "trajetoria_xy_m": caminho.round(5).tolist(),
            "trajetoria_xytheta": [
                [round(float(x), 5), round(float(y), 5), 0.0]
                for x, y in caminho
            ],
        }, f, indent=4)
    print(f"[OK] Trajetoria salva em '{args.saida}'")

    grid = None
    if args.metodo == "grid":
        livre, origem = construir_grid(expandidos, limites_c, args.passo)
        objetivo_ij = _celula(args.objetivo, origem, args.passo, livre.shape)
        objetivo_ij = _celula_livre_mais_proxima(objetivo_ij, livre)
        valores = (mapa_wavefront(livre, objetivo_ij)
                   if objetivo_ij is not None else None)
        grid = (livre, origem, args.passo, valores)

    try:
        if grid is not None and valores is not None and livre.size <= 4000:
            plotar_valores(caminho, obstaculos, args.inicio, args.objetivo,
                           grid, args.figura)
        else:
            if grid is not None:
                print(f"[AVISO] {livre.size} celulas: numeros ilegiveis, "
                      f"desenhando o campo em cores (use --passo maior "
                      f"para a tabela de valores)")
            plotar(caminho, obstaculos, expandidos, args.inicio,
                   args.objetivo, limites, args.figura, grid=grid)
    except ImportError:
        try:
            plotar_opencv(caminho, obstaculos, expandidos, args.inicio,
                          args.objetivo, limites, args.figura, grid=grid)
        except Exception as e:
            print(f"[AVISO] matplotlib indisponivel e falha no OpenCV ({e}); figura nao gerada")

    # Envio para o robo usando exclusivamente a odometria interna (protocolo demo.py)
    if sucesso and ser is not None:
        try:
            waypoints = caminho[1:].copy()
            if len(waypoints) > MAX_WAYPOINTS_ROBO:
                waypoints = waypoints[:MAX_WAYPOINTS_ROBO]

            theta_inicial_deg = robo.get("theta_graus") if robo else None
            if theta_inicial_deg is None and robo and "theta_rad" in robo:
                theta_inicial_deg = float(np.degrees(robo["theta_rad"]))
            if theta_inicial_deg is None:
                d0_vec = caminho[1] - caminho[0] if len(caminho) > 1 else np.zeros(2)
                theta_inicial_deg = float(np.degrees(np.arctan2(d0_vec[1], d0_vec[0])))

            rx = float(robo.get("x_m")) if robo else float(caminho[0, 0])
            ry = float(robo.get("y_m")) if robo else float(caminho[0, 1])
            pos_ini = np.array([rx, ry], dtype=float)

            # Pre-compensacao de deriva e escala da odometria interna
            waypoints_cmd = compensar_deriva_odometria(
                pos_ini, waypoints,
                fator_escala=args.fator_odometria,
                comp_angular_graus=args.comp_angular_graus
            )
            if abs(args.fator_odometria - 1.0) > 1e-6 or abs(args.comp_angular_graus) > 1e-6:
                print(f"[INFO] Compensacao de odometria aplicada: "
                      f"fator={args.fator_odometria:.3f}, "
                      f"ang={args.comp_angular_graus:+.1f}°")

            # Se o firmware usar eixo Y oposto ao do mapa, espelha dy e o sinal de theta
            if args.inverter_y_robo:
                y_max = float(limites.get("y_max", 1.21))
                ry_cmd = y_max - ry
                theta_cmd_deg = -float(theta_inicial_deg)
                waypoints_cmd[:, 1] = y_max - waypoints_cmd[:, 1]
                print("[INFO] Eixo Y e sinal de theta invertidos para o referencial do robo.")
            else:
                ry_cmd = ry
                theta_cmd_deg = float(theta_inicial_deg)

            print(f"[OK] Enviando pose inicial e {len(waypoints_cmd)} waypoints "
                  f"(maximo permitido: {MAX_WAYPOINTS_ROBO})...")

            # 1. Para e limpa lista anterior (<C>)
            stop(ser)

            # 2. Define pose inicial completa (x, y, theta_inicial em graus) via <O,x,y,th>
            resp_pose = set_pose(ser, rx, ry_cmd, theta_cmd_deg)
            print(f"[INFO] Pose inicial definida: <O,{rx:.3f},{ry_cmd:.3f},{theta_cmd_deg:.1f}> -> {resp_pose}")

            # 3. Envia lista de waypoints com theta = 0.0 via <W,n,x0,y0,0.0,...>
            pts_envio = [(float(wp[0]), float(wp[1]), 0.0) for wp in waypoints_cmd]
            resp_wp = set_waypoints(ser, pts_envio)
            if resp_wp is None or not resp_wp.startswith("<OK"):
                # Fallback: adiciona um por um via <A,x,y,0.0>
                stop(ser)
                set_pose(ser, rx, ry_cmd, theta_cmd_deg)
                for i, (wx, wy, wth) in enumerate(pts_envio, 1):
                    r_add = add_waypoint(ser, wx, wy, wth)
                    print(f"  [WP {i:2d}/{len(pts_envio)}] ({wx:.3f}, {wy:.3f}, {wth:.1f}) -> {r_add}")
            else:
                for i, (wx, wy, wth) in enumerate(pts_envio, 1):
                    print(f"  [WP {i:2d}/{len(pts_envio)}] ({wx:.3f}, {wy:.3f}, {wth:.1f})")
                print(f"[INFO] Lista enviada via <W>: {resp_wp}")

            # 4. Inicia a navegacao (<R>) e monitora telemetria interna (<T>)
            ok_st, resp_st, msg_st = start(ser, wait_auto_eq=True)
            print(f"[INFO] Início da navegação: {resp_st} ({msg_st})")
            if ok_st and not args.sem_telemetria:
                print("[INFO] Monitorando odometria interna via <T> até concluir (Ctrl+C para parar)...")
                try:
                    done, last_t = wait_until_done_telemetry(ser)
                    if done and last_t is not None:
                        print(f"[OK] Robô concluiu navegação! Pose final pela odometria interna: "
                              f"({last_t['x']:.3f}m, {last_t['y']:.3f}m, {last_t['th_deg']:.1f}°)")
                except KeyboardInterrupt:
                    print("\n[AVISO] Monitoramento interrompido pelo usuário. Enviando <C> (stop)...")
                    stop(ser)
        finally:
            ser.close()


if __name__ == "__main__":
    main()
