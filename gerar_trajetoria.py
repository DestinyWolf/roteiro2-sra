"""Geracao de trajetoria por decomposicao em celulas (grid).

Le o mapa de obstaculos poligonais produzido por 'gerar_mapa.py'
(mapa_obstaculos.json) e gera uma trajetoria do ponto inicial ate o
objetivo.

O metodo padrao e o de grid: o workspace e discretizado numa grade
regular, cada celula e classificada em livre ou ocupada, e o caminho
sai de uma busca A* 8-conectada sobre as celulas livres, depois
suavizada por atalhos em linha de visada. A busca e completa e otima
na resolucao da grade: se existe caminho representavel, ela o acha.

O metodo de campos potenciais (descida de gradiente sobre
U = U_attr + U_rep) continua disponivel em '--metodo potencial', para
comparacao: e reativo e barato, mas nao e completo (pode parar num
minimo local).

O robo e um disco de raio '--raio-robo' (0.22 m por padrao). O
planejamento acontece no espaco de configuracoes: 'expandir_obstaculos'
infla cada obstaculo pela soma de Minkowski com um disco desse raio, e
sobre o mapa inflado o robo volta a ser um ponto. Assim um ponto livre
no mapa inflado equivale ao disco inteiro livre no mapa real.

'--margem' soma uma folga extra ao raio; o raio usado na expansao e
'raio_robo + margem'. Com os dois em zero o planejamento e o do ponto
puro.

A trajetoria entregue tem no maximo MAX_PONTOS_TRAJETORIA pontos (10),
contando o inicio e o objetivo -- ou seja, no maximo 9 waypoints para o
robo, que ja esta no primeiro ponto. O limite vale para os dois metodos
e e ajustavel por '--max-pontos'.

Quem corta e 'limitar_pontos': remove um vertice de cada vez, sempre o
que menos altera a forma do caminho, e so aceita a remocao quando o
atalho que substitui o vertice continua em linha de visada livre. Por
isso a trajetoria de 10 pontos ainda e um caminho valido, e nao uma
aproximacao do original. No outro sentido, '--min-pontos' (10) completa
com pontos dentro dos segmentos quando o caminho sai curto demais,
sempre mantendo os vertices, para o controlador receber setpoints bem
distribuidos.

'orientacoes' calcula o theta de cada ponto (a direcao do segmento que
sai dele) e o grava no JSON.
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

DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 57600
TIMEOUT_S = 1.0
RESPONSE_TIMEOUT_S = 2.0
AUTO_EQ_S = 2.0
START_GRACE_S = 2.5

STATE_NAMES = {0: "GIRAR", 1: "ANDAR", 2: "DONE"}

# Numero de pontos da trajetoria, contando o inicio e o objetivo. O
# robo recebe um waypoint por ponto menos o inicio, que e a pose onde
# ele ja esta: no limite, 9 waypoints.
# IMPORTANTE: MIN_PONTOS_TRAJETORIA = 0 evita subdividir trechos retos em
# pontos intermediarios colineares (cada parada intermediaria faz o firmware
# trocar ANDAR -> GIRAR -> ANDAR, dando um "coice" que derrapa as rodas e
# acumula deriva angular na odometria).
MAX_PONTOS_TRAJETORIA = 10
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
    soma de Minkowski dos dois. Fazer isso sobre uma mascara rasterizada
    (e nao deslocando as arestas uma a uma) resolve de graca os dois
    casos em que o offset analitico falha: vertices reflexos, onde as
    arestas deslocadas se auto-interceptam, e obstaculos que se fundem
    ao crescer.

    'margem' aqui e o raio total inflado: o raio do robo mais a folga
    de seguranca pedida. Com margem 0 a funcao devolve os obstaculos
    inalterados.

    Pode devolver menos poligonos que a entrada, quando dois obstaculos
    se encostam apos a expansao.
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
        # Simplifica mantendo o erro bem abaixo da resolucao do raster.
        c = cv2.approxPolyDP(c, 0.8, True)   # 0.8 pixel
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
    # O contorno real da mesa (fita preta) tambem encolhe: o robo, de raio
    # 'margem', precisa ficar inteiro dentro dele.
    contorno = limites_mesa.contorno_de_limites(limites)
    if contorno is not None:
        encolhidos["contorno"] = limites_mesa.encolher_contorno(contorno, margem)
    return encolhidos


# ---------------------------------------------------------------- potencial

def potencial(q, objetivo, obstaculos, k_attr=1.0, k_rep=0.001, d0=0.08,
              d_max=0.3, limites=None):
    """Potencial total e seu gradiente em q.

    Atrativo: quadratico perto do objetivo e conico acima de 'd_max',
    para evitar gradientes muito grandes quando o objetivo esta longe.
    Repulsivo: potencial de Khatib, ativo somente a menos de 'd0' do
    obstaculo (e tambem das paredes do workspace), multiplicado por ||q - objetivo||^2 para que se anule no
    objetivo (correcao GNRON, necessaria quando o objetivo esta a menos
    de 'd0' de um obstaculo).
    """
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

    # (distancia, direcao de afastamento) de cada fonte repulsiva
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

        # Contorno real da mesa (fita preta): age como parede repulsiva.
        contorno = limites_mesa.contorno_de_limites(limites)
        if contorno is not None:
            p = limites_mesa.ponto_mais_proximo_contorno(q, contorno)
            dist = float(np.linalg.norm(q - p))
            fora = not limites_mesa.dentro_varios(q, contorno)[0]
            d = -dist if fora else dist
            if d < d0 and dist > 1e-9:
                # 'direcao' aponta da parede para q (para dentro da mesa
                # quando q esta dentro; para fora quando q escapou).
                fontes.append((d, (q - p) / dist))

    for d, direcao in fontes:
        if d <= 0.0:
            # Dentro do obstaculo/parede: empurra de volta com forca saturada.
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
    """Descida de gradiente sobre o potencial.

    Campos potenciais tem minimos locais; quando a distancia ao objetivo
    deixa de cair, o planejador passa a seguir a direcao tangencial ao
    gradiente (contorno do obstaculo) por alguns passos ate voltar a
    progredir. Retorna (caminho, sucesso).
    """
    limites = kw_potencial.get("limites")
    q = np.array(inicio, dtype=float)
    objetivo = np.array(objetivo, dtype=float)
    caminho = [q.copy()]

    def confinar(p):
        """Mantem o ponto dentro do workspace."""
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
        """Verdadeiro se p esta fora de todos os obstaculos."""
        return all(dist_ponto_poligono(p, o) > 0.0 for o in obstaculos)

    def sentido_contorno(p, direcao_grad):
        """Lado do contorno cuja tangente aponta mais para o objetivo."""
        perp = np.array([-direcao_grad[1], direcao_grad[0]])
        return 1.0 if perp @ (objetivo - p) >= 0.0 else -1.0

    janela = 50            # passos entre duas medidas de progresso
    passos_tangencial = 0  # passos restantes em modo de contorno
    sinal = 1.0            # sentido do contorno
    tentativas = 0         # escapes consecutivos sem progresso
    escapes = 0
    d_anterior = np.linalg.norm(q - objetivo)

    for it in range(max_iter):
        if np.linalg.norm(q - objetivo) < tol:
            return np.array(caminho), True

        _, grad = potencial(q, objetivo, obstaculos, **kw_potencial)
        norma = np.linalg.norm(grad)

        # Progresso medido a cada 'janela' passos.
        if it > 0 and it % janela == 0:
            d_atual = np.linalg.norm(q - objetivo)
            if d_atual > d_anterior - 1e-4 and passos_tangencial == 0:
                if escapes >= max_escapes:
                    break  # minimo local do qual nao se escapou
                escapes += 1
                # Primeira tentativa: contorna pelo lado do objetivo.
                # Se nao resolveu, tenta o outro lado, indo mais longe.
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
            # Minimo local exato (gradiente nulo): escapa pela tangente
            # a direcao do objetivo.
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
            # Gira a direcao 90 graus: segue o contorno do obstaculo.
            direcao = sinal * np.array([-direcao[1], direcao[0]])
            passos_tangencial -= 1

        # Nunca entra no obstaculo: tenta a direcao do gradiente e, se
        # ela estiver bloqueada, as duas tangentes e passos menores.
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
            break  # totalmente bloqueado

        q = q_novo
        caminho.append(q.copy())

    sucesso = np.linalg.norm(q - objetivo) < tol
    return np.array(caminho), sucesso


# -------------------------------------------------------------------- grid

def construir_grid(obstaculos, limites, passo):
    """Rasteriza o workspace numa grade regular de ocupacao.

    Devolve (livre, origem): 'livre' e uma matriz booleana
    [altura x largura] com True nas celulas cujo centro esta fora de
    todos os obstaculos, e 'origem' e o canto (x_min, y_min). O centro
    da celula (i, j) fica em origem + (j + 0.5, i + 0.5) * passo.

    A classificacao olha apenas o centro da celula, entao uma celula
    parcialmente ocupada pode sair livre. Isso e o erro de
    discretizacao inerente ao metodo; a folga em relacao aos
    obstaculos vem de '--margem', aplicada antes (Minkowski).
    """
    x0, y0 = limites["x_min"], limites["y_min"]
    largura = max(1, int(np.ceil((limites["x_max"] - x0) / passo)))
    altura = max(1, int(np.ceil((limites["y_max"] - y0) / passo)))
    origem = np.array([x0, y0], dtype=float)

    ocupado = np.zeros((altura, largura), dtype=np.uint8)
    try:
        import cv2
    except ImportError:
        # Sem OpenCV: testa o centro de cada celula por ray casting.
        for i in range(altura):
            for j in range(largura):
                c = origem + (np.array([j, i]) + 0.5) * passo
                if any(ponto_dentro_poligono(c, o) for o in obstaculos):
                    ocupado[i, j] = 1
    else:
        # fillPoly trabalha em indices de celula; o centro da celula j
        # corresponde ao vertice x em (x - x0)/passo - 0.5.
        for poligono in obstaculos:
            pts = np.round((np.asarray(poligono, dtype=float) - origem)
                           / passo - 0.5).astype(np.int32)
            cv2.fillPoly(ocupado, [pts], 1)

    # Fora do contorno real da mesa (fita preta) a celula e proibida.
    contorno = limites_mesa.contorno_de_limites(limites)
    if contorno is not None:
        xs = origem[0] + (np.arange(largura) + 0.5) * passo
        ys = origem[1] + (np.arange(altura) + 0.5) * passo
        gx, gy = np.meshgrid(xs, ys)
        centros = np.column_stack([gx.ravel(), gy.ravel()])
        dentro = limites_mesa.dentro_varios(centros, contorno).reshape(altura, largura)
        ocupado[~dentro] = 1

    return ocupado == 0, origem


def mapa_wavefront(livre, objetivo_ij):
    """Valor de cada celula: numero de passos ate o objetivo (NF1).

    Busca em largura 8-conectada a partir do objetivo, sem cortar canto
    de celula ocupada -- a mesma conectividade da busca A*. O objetivo
    vale 0 e cada anel seguinte vale um a mais; descer esse campo celula
    a celula leva ao objetivo, e por isso ele e o 'potencial discreto'
    do metodo de grid, sem minimos locais espurios.

    Devolve uma matriz int com -1 nas celulas ocupadas ou inalcancaveis.
    """
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
                        continue  # nao corta canto
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


def busca_a_estrela(livre, inicio_ij, objetivo_ij):
    """A* 8-conectado sobre as celulas livres.

    Custo 1 nos passos ortogonais e sqrt(2) nos diagonais, com a
    distancia octil como heuristica -- admissivel e consistente, logo o
    caminho em celulas e otimo. Diagonais que cortariam o canto de uma
    celula ocupada sao proibidas.

    Devolve a lista de celulas [(i, j), ...] ou None se nao ha caminho.
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
                continue  # nao corta canto
            novo = custo + peso
            if novo < g.get(v, np.inf):
                g[v] = novo
                veio_de[v] = atual
                heapq.heappush(fronteira, (novo + h(v), novo, v))
    return None


def _visada_livre(a, b, livre, origem, passo):
    """Verdadeiro se o segmento ab so atravessa celulas livres."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = int(np.ceil(np.linalg.norm(b - a) / (0.5 * passo))) + 1
    for t in np.linspace(0.0, 1.0, n):
        c = _celula(a + t * (b - a), origem, passo, livre.shape)
        if not livre[c]:
            return False
    return True


def suavizar_caminho(pontos, livre, origem, passo):
    """Remove os degraus do caminho por atalhos em linha de visada.

    Avanca pelo caminho ligando cada ponto ao vertice mais distante que
    ainda alcanca em linha reta sem cruzar celula ocupada. Encurta o
    caminho e tira o vies das 8 direcoes da grade.
    """
    if len(pontos) <= 2:
        return np.asarray(pontos, dtype=float)

    saida = [pontos[0]]
    i = 0
    while i < len(pontos) - 1:
        j = len(pontos) - 1
        while j > i + 1 and not _visada_livre(pontos[i], pontos[j],
                                              livre, origem, passo):
            j -= 1
        saida.append(pontos[j])
        i = j
    return np.asarray(saida, dtype=float)


def reamostrar(pontos, passo_amostra):
    """Reamostra a poligonal com espacamento aproximadamente uniforme."""
    pontos = np.asarray(pontos, dtype=float)
    if len(pontos) < 2:
        return pontos
    saida = [pontos[0]]
    for a, b in zip(pontos[:-1], pontos[1:]):
        d = np.linalg.norm(b - a)
        n = max(1, int(np.ceil(d / passo_amostra)))
        for k in range(1, n + 1):
            saida.append(a + (b - a) * (k / n))
    return np.asarray(saida)


def _dentro_poligono_varios(pontos, poligono):
    """Ray casting vetorizado: mascara dos pontos dentro do poligono.

    Mesma regra de 'ponto_dentro_poligono', so que para um array de
    pontos de uma vez -- o teste roda milhares de vezes dentro de
    'limitar_pontos' e em Python puro ficaria lento.
    """
    x, y = pontos[:, 0], pontos[:, 1]
    dentro = np.zeros(len(pontos), dtype=bool)
    n = len(poligono)
    for i in range(n):
        x1, y1 = poligono[i]
        x2, y2 = poligono[(i + 1) % n]
        if y1 == y2:
            continue  # aresta horizontal nunca cruza o raio
        cruza = (y1 > y) != (y2 > y)
        x_cruz = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        dentro ^= cruza & (x < x_cruz)
    return dentro


def _visada_poligonos(a, b, obstaculos, limites=None, passo=0.005):
    """'_visada_livre' sem grade: testa os poligonos direto.

    Usada para limitar os pontos do caminho do metodo potencial, que
    nao constroi grade nenhuma.
    """
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
    """Reduz a poligonal a no maximo 'maximo' pontos, sem colidir.

    Remove um vertice por vez -- sempre o que menos altera a forma do
    caminho, isto e, o mais proximo do atalho que o substituiria -- e so
    aceita a remocao quando esse atalho continua em linha de visada
    livre. O inicio e o objetivo nunca saem.

    E o criterio do Ramer-Douglas-Peucker com o teste de colisao no
    meio: por isso o resultado nao e apenas parecido com o caminho
    original, e um caminho valido.

    Devolve (pontos, cabe). 'cabe' e False quando nao houve como chegar
    a 'maximo' pontos sem cortar obstaculo; nesse caso o retorno e a
    menor poligonal segura encontrada, e quem chama avisa.
    """
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
        """Distancia do vertice k ao atalho que o substituiria."""
        a, b, c = pontos[anterior[k]], pontos[k], pontos[seguinte[k]]
        return float(np.linalg.norm(b - ponto_mais_proximo_segmento(b, a, c)))

    fila = [(desvio(k), 0, k) for k in range(1, n - 1)]
    heapq.heapify(fila)

    while restantes > maximo and fila:
        _, v, k = heapq.heappop(fila)
        if not vivo[k] or v != versao[k]:
            continue  # entrada velha: os vizinhos de k mudaram depois
        a, c = anterior[k], seguinte[k]
        if not visada_livre(pontos[a], pontos[c]):
            continue  # o atalho bateria no obstaculo; tenta o proximo
        vivo[k] = False
        restantes -= 1
        seguinte[a] = c
        anterior[c] = a
        # Os vizinhos passaram a ter outro atalho: reavalia os dois,
        # inclusive se um deles ja tinha sido recusado antes.
        for m in (a, c):
            if 0 < m < n - 1 and vivo[m]:
                versao[m] += 1
                heapq.heappush(fila, (desvio(m), versao[m], m))

    saida = np.asarray([p for p, ok in zip(pontos, vivo) if ok], dtype=float)
    return saida, len(saida) <= maximo


def reamostrar_n(pontos, n):
    """Reamostra a poligonal em exatamente n pontos.

    Todos os vertices originais sao mantidos e os pontos que faltam
    para chegar a n vao para dentro dos segmentos, os mais longos
    primeiro. Manter os vertices e o que garante que o caminho
    continua livre de colisao: amostrando so por comprimento de arco,
    as quinas ficariam de fora, e a reta entre as duas amostras de
    cada lado de uma quina corta o canto -- podendo entrar justamente
    no obstaculo que a quina contornava.
    """
    pontos = np.asarray(pontos, dtype=float)
    if len(pontos) < 2 or n <= len(pontos):
        return pontos

    distancias = np.linalg.norm(np.diff(pontos, axis=0), axis=1)
    total = float(distancias.sum())
    if total <= 0.0:
        return pontos

    # Reparticao proporcional ao comprimento de cada segmento, pelo
    # metodo dos maiores restos, para a soma bater exatamente com o
    # numero de pontos que faltam.
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
                   maximo=MAX_PONTOS_TRAJETORIA):
    """Deixa a trajetoria com um numero de waypoints entre os limites.

    Acima do maximo, 'limitar_pontos' tira os vertices um a um, e so
    aceita tirar quando o atalho que substitui o vertice continua em
    celulas livres -- por isso a reducao nunca introduz colisao.
    Abaixo do minimo, 'reamostrar_n' acrescenta pontos dentro dos
    proprios segmentos ate o alvo.
    """
    pontos = np.asarray(pontos, dtype=float)

    if maximo and len(pontos) > maximo:
        antes = len(pontos)
        pontos, cabe = limitar_pontos(
            pontos, maximo,
            lambda a, b: _visada_livre(a, b, livre, origem, passo))
        print(f"[OK] Trajetoria reduzida de {antes} para {len(pontos)} "
              f"waypoints (limite {maximo})")
        if not cabe:
            print(f"[AVISO] Nao da para descer de {len(pontos)} waypoints "
                  f"sem cortar obstaculo")

    if minimo and len(pontos) < minimo:
        pontos = reamostrar_n(pontos, minimo)
    return pontos


def orientacoes(pontos, theta_final=None):
    """Theta de cada waypoint: a direcao do segmento que sai dele.

    O robo e um disco, entao a orientacao nao entra no planejamento;
    ela e so a direcao de deslocamento que o controlador deve seguir.
    O ultimo ponto herda a orientacao do segmento que chega nele, ou
    'theta_final' se for dado.
    """
    pontos = np.asarray(pontos, dtype=float)
    if len(pontos) < 2:
        return np.zeros(len(pontos))
    d = np.diff(pontos, axis=0)
    thetas = np.arctan2(d[:, 1], d[:, 0])
    ultimo = thetas[-1] if theta_final is None else theta_final
    return np.concatenate([thetas, [ultimo]])


def plan_grid(inicio, objetivo, obstaculos, limites, passo=0.01,
              suavizar=True, min_pontos=MIN_PONTOS_TRAJETORIA,
              max_pontos=MAX_PONTOS_TRAJETORIA):
    """Planejamento por decomposicao em celulas e busca A*.

    Discretiza o workspace, acha o caminho otimo em celulas, suaviza-o
    por linha de visada, reduz a no maximo 'max_pontos' pontos e
    reamostra com espacamento uniforme dentro desse mesmo limite.
    Retorna (caminho, sucesso); sucesso e False somente quando nao
    existe caminho na resolucao pedida.
    """
    livre, origem = construir_grid(obstaculos, limites, passo)

    inicio_ij = _celula(inicio, origem, passo, livre.shape)
    objetivo_ij = _celula(objetivo, origem, passo, livre.shape)

    # O ponto pode cair numa celula marcada ocupada pelo arredondamento
    # da rasterizacao; nesse caso parte da celula livre mais proxima.
    for nome, c in (("inicio", inicio_ij), ("objetivo", objetivo_ij)):
        if not livre[c]:
            novo = _celula_livre_mais_proxima(c, livre)
            # So vale corrigir o arredondamento da rasterizacao: se a
            # celula livre mais proxima esta longe, o ponto pedido e
            # realmente inalcancavel e mover o objetivo seria resolver
            # outro problema sem avisar.
            if novo is None or max(abs(novo[0] - c[0]),
                                   abs(novo[1] - c[1])) > 1:
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

    celulas = busca_a_estrela(livre, inicio_ij, objetivo_ij)
    n_livres = int(livre.sum())
    print(f"[OK] Grade {livre.shape[1]}x{livre.shape[0]} celulas de "
          f"{passo} m ({n_livres} livres, "
          f"{livre.size - n_livres} ocupadas)")
    if celulas is None:
        print("[AVISO] Nao existe caminho na resolucao pedida "
              "(tente --passo menor ou --margem menor)")
        return np.array([np.asarray(inicio, dtype=float)]), False

    # Pontos reais: inicio exato, centros das celulas, objetivo exato.
    pontos = [np.asarray(inicio, dtype=float)]
    pontos += [_centro(c, origem, passo) for c in celulas]
    pontos.append(np.asarray(objetivo, dtype=float))

    if suavizar:
        pontos = suavizar_caminho(pontos, livre, origem, passo)
    pontos = np.asarray(pontos, dtype=float)

    if max_pontos or min_pontos:
        pontos = reduzir_pontos(pontos, livre, origem, passo,
                                minimo=min_pontos, maximo=max_pontos)
    return pontos, True


# ------------------------------------------------------------------- mapa

def carregar_mapa(caminho_json):
    """Le obstaculos, limites e pose do robo do JSON de 'gerar_mapa.py'.

    Devolve (obstaculos, limites, robo). 'robo' e o dicionario do bloco
    'robo' do JSON -- a pose medida pela camera, usada como ponto
    inicial -- ou None quando o mapa nao traz esse bloco.
    """
    with open(caminho_json, "r", encoding="utf-8") as f:
        dados = json.load(f)

    obstaculos = [
        np.array(obs["vertices_xy_m"], dtype=float)
        for obs in dados.get("obstaculos", [])
    ]
    limites = dados.get("workspace_limites_m", {})

    # Contorno real da mesa (interior da fita preta), medido na calibracao:
    # os planejadores nao deixam o robo sair dele.
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
    """Desenha a grade como tabela: o valor escrito dentro de cada celula.

    Celula ocupada (ou inalcancavel a partir do objetivo) aparece como
    'Inf', como e usual no wavefront. As celulas por onde a trajetoria
    passa ficam realcadas, e as do inicio e do objetivo tambem.

    So faz sentido numa grade grossa; com celulas pequenas os numeros
    nao cabem e 'plotar' desenha o campo em cores.
    """
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

    # Contorno real dos obstaculos sobre a grade, para ver o quanto a
    # discretizacao distorce a geometria.
    for poligono in obstaculos:
        ax.add_patch(Polygon(poligono, closed=True, fill=False,
                             edgecolor="k", lw=1.6, zorder=2))

    ax.plot(caminho[:, 0], caminho[:, 1], "-b", lw=2.0, zorder=3)
    ax.plot(*inicio, "go", ms=9, mec="k", zorder=4)
    ax.plot(*objetivo, "r*", ms=15, mec="k", zorder=4)

    ax.set_xlim(origem[0], origem[0] + largura * passo)
    ax.set_ylim(origem[1], origem[1] + altura * passo)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"valores da grade (wavefront) - celula de {passo} m",
                 fontsize=10)
    fig.savefig(arquivo, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figura salva em '{arquivo}'")


def plotar(caminho, obstaculos, expandidos, inicio, objetivo, limites,
           arquivo, grid=None):
    """Desenha mapa e trajetoria; com 'grid', mostra os pontos da grade.

    'grid' e a tupla (livre, origem, passo, valores): cada celula vira
    um ponto no seu centro, azul se livre e vermelho se ocupada, e
    'valores' (wavefront, ou None) pinta o campo de fundo e escreve o
    valor em cada celula quando a grade e grossa o bastante.
    """
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
            # Campo de valores (wavefront) como fundo; as celulas
            # ocupadas ou inalcancaveis ficam transparentes.
            campo = np.ma.masked_less(valores, 0)
            imagem = ax.imshow(
                campo, origin="lower", zorder=-1, cmap="viridis", alpha=0.75,
                extent=(origem[0], origem[0] + largura * passo,
                        origem[1], origem[1] + altura * passo))
            barra = fig.colorbar(imagem, ax=ax, fraction=0.046, pad=0.04)
            barra.set_label("valor da celula (passos ate o objetivo)",
                            fontsize=8)
            barra.ax.tick_params(labelsize=7)

            # O numero so cabe quando a grade e grossa.
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
                label="margem de seguranca" if i == 0 else None))
    for i, poligono in enumerate(obstaculos):
        # Com a grade visivel, o preenchimento fica translucido para
        # nao esconder as celulas ocupadas.
        ax.add_patch(Polygon(poligono, closed=True, facecolor="0.6",
                             edgecolor="k",
                             alpha=0.45 if grid is not None else 1.0,
                             label="obstaculo" if i == 0 else None))
    ax.plot(caminho[:, 0], caminho[:, 1], "-b", lw=2, label="trajetoria")
    ax.plot(*inicio, "go", ms=10, label="inicio")
    ax.plot(*objetivo, "r*", ms=15, label="objetivo")
    contorno_mesa = limites_mesa.contorno_de_limites(limites)
    if contorno_mesa is not None:
        ax.add_patch(Polygon(contorno_mesa, closed=True, fill=False,
                             edgecolor="m", lw=2, label="limite da mesa"))
    ax.set_xlim(limites.get("x_min", 0.0), limites.get("x_max", 1.2))
    ax.set_ylim(limites.get("y_min", 0.0), limites.get("y_max", 0.8))
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if grid is not None:
        # O ponto so fica legivel se for menor que o espacamento real
        # da grade na figura, que depende do tamanho dos eixos.
        fig.canvas.draw()
        caixa = ax.get_window_extent()
        pts_por_m = (caixa.height / (ax.get_ylim()[1] - ax.get_ylim()[0])
                     * 72.0 / fig.dpi)
        for artista in pontos_grid:
            artista.set_markersize(max(0.3, 0.45 * pts_por_m * passo))

    legenda = ax.legend(loc="best", fontsize=8)
    if grid is not None:
        # Na legenda os mesmos pontos precisam ser visiveis.
        handles = getattr(legenda, "legend_handles",
                          getattr(legenda, "legendHandles", []))
        for handle in handles[:len(pontos_grid)]:
            handle.set_markersize(7)
    ax.grid(True, alpha=0.3)
    fig.savefig(arquivo, dpi=120, bbox_inches="tight")
    print(f"[OK] Figura salva em '{arquivo}'")



#-------------------------------------------------- Comandos do ROBO

# ---------------------------------------------------------------------------
# Conexão
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Envio e leitura
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Waypoints
# ---------------------------------------------------------------------------
def add_waypoint(ser, x, y, th=None):
    if th is not None:
        return cmd_and_wait(ser, f"A,{x:.3f},{y:.3f},{th:.1f}")
    return cmd_and_wait(ser, f"A,{x:.3f},{y:.3f}")


def set_waypoints(ser, pts):
    if not pts:
        return cmd_and_wait(ser, "C")
    body = f"W,{len(pts)}"
    for pt in pts:
        if len(pt) >= 3:
            body += f",{pt[0]:.3f},{pt[1]:.3f},{pt[2]:.1f}"
        else:
            body += f",{pt[0]:.3f},{pt[1]:.3f}"
    return cmd_and_wait(ser, body)


def append_waypoints(ser, pts):
    if not pts:
        return None
    body = f"Q,{len(pts)}"
    for pt in pts:
        if len(pt) >= 3:
            body += f",{pt[0]:.3f},{pt[1]:.3f},{pt[2]:.1f}"
        else:
            body += f",{pt[0]:.3f},{pt[1]:.3f}"
    return cmd_and_wait(ser, body)


def list_waypoints(ser):
    resp = cmd_and_wait(ser, "L")
    if resp is None or not resp.startswith("<L,"):
        return None
    parts = resp[1:-1].split(",")[1:]
    n = int(parts[0])
    wps = []
    for i in range(n):
        x = float(parts[1 + 2 * i])
        y = float(parts[2 + 2 * i])
        wps.append((x, y))
    return wps


def delete_waypoint(ser, idx):
    return cmd_and_wait(ser, f"D,{idx}")


def replace_waypoint(ser, idx, x, y):
    return cmd_and_wait(ser, f"S,{idx},{x:.3f},{y:.3f}")

# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
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


def wait_until_done_telemetry(ser, poll_s=0.4, timeout_s=90.0):
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


def detectar_pose_camera(cam_arg, calib_arq="calibracao_mesa.json"):
    """Captura um frame da câmera e retorna o dicionário 'dados_robo' atualizado em metros."""
    import gerar_mapa as gm
    cfg = gm.ConfigMapa(sem_gui=True)
    gm.carregar_calibracao(calib_arq, cfg)
    frame = gm.capturar_frame(cam_arg)
    if cfg.cantos_calibrados is not None:
        pontos_origem = np.asarray(cfg.cantos_calibrados, dtype=np.float32)
    else:
        pontos_origem = gm.detectar_mesa_auto(frame)
    mesa_ret, H_mesa = gm.retificar_perspectiva(frame, pontos_origem, cfg)
    mascara_mesa, _ = gm.construir_mascara_mesa_retificada(H_mesa, mesa_ret.shape[:2], cfg)
    dados_robo, _ = gm.detectar_robo_hsv(mesa_ret, cfg, mascara_mesa)
    return dados_robo

def main():
    ap = argparse.ArgumentParser(
        description="Gera trajetoria por decomposicao em celulas (grid) "
                    "ou por campos potenciais.")
    ap.add_argument("--mapa", default="mapa_obstaculos.json")
    # Por padrao o inicio e a pose do robo medida pela camera, gravada
    # no bloco 'robo' do mapa; --inicio sobrescreve.
    ap.add_argument("--inicio", type=float, nargs=2, default=None,
                    help="ponto inicial [m]; o padrao e a pose do robo "
                         "lida do mapa")
    # (1.1, 0.7) do mapa antigo caiu dentro de um obstaculo no
    # mapa atual (workspace 1.2 x 0.8 m).
    ap.add_argument("--objetivo", type=float, nargs=2, default=[1.1, 0.4])
    ap.add_argument("--metodo", choices=("grid", "potencial"), default="grid",
                    help="grid: A* sobre grade de ocupacao (padrao); "
                         "potencial: descida de gradiente")
    ap.add_argument("--passo", type=float, default=0.02,
                    help="[grid] lado da celula da grade [m]")
    ap.add_argument("--sem-suavizar", action="store_true",
                    help="[grid] mantem o caminho em degraus da grade")
    ap.add_argument("--min-pontos", type=int,
                    default=MIN_PONTOS_TRAJETORIA,
                    help=f"[grid] minimo de pontos da trajetoria (padrao "
                         f"{MIN_PONTOS_TRAJETORIA}); 0 nao completa")
    ap.add_argument("--max-pontos", type=int,
                    default=MAX_PONTOS_TRAJETORIA,
                    help=f"maximo de pontos da trajetoria, contando o "
                         f"inicio e o objetivo (padrao "
                         f"{MAX_PONTOS_TRAJETORIA}); 0 remove o limite")
    ap.add_argument("--k-attr", type=float, default=1.0)
    ap.add_argument("--k-rep", type=float, default=0.001)
    ap.add_argument("--d0", type=float, default=0.08,
                    help="raio de influencia do campo repulsivo [m]")
    ap.add_argument("--raio-robo", type=float, default=0.14,
                    help="raio do robo [m]; infla os obstaculos pela soma "
                         "de Minkowski (padrao 0.14 m = largura real das rodas + folga)")
    ap.add_argument("--margem", type=float, default=0.0,
                    help="folga de seguranca [m] somada ao raio do robo")
    ap.add_argument("--fator-odometria", type=float, default=1.0,
                    help="fator de escala aplicado aos deslocamentos enviados ao robo "
                         "(ex: 0.85 se o robo anda ~18%% a mais do que o comandado)")
    ap.add_argument("--sem-telemetria", action="store_true",
                    help="nao aguarda/monitora a telemetria <T> apos iniciar a navegacao")
    ap.add_argument("--malha-fechada", action="store_true",
                    help="envia 1 waypoint por vez e corrige a pose (x, y, theta) via camera (--cam) a cada parada")
    ap.add_argument("--cam", type=str, nargs="?", const="Iriun", default=None,
                    help="camera usada no modo --malha-fechada (padrao: Iriun ou indice como 0)")
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--saida", default="trajetoria.json")
    ap.add_argument("--figura", default="trajetoria.png")
    ap.add_argument("--sem-envio", action="store_true",
                    help="so gera a trajetoria e a figura, sem abrir a "
                         "porta serial nem enviar waypoints")
    ap.add_argument("--port", default=DEFAULT_PORT, help="Porta serial utilizada para a comunicação com a placa")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Baudrate utilizado  para a comunicação com a placa")

    args = ap.parse_args()

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

    # Espaco de configuracoes: obstaculos inflados pelo raio do robo
    # (mais a folga pedida), para o robo voltar a ser um ponto.
    raio = args.raio_robo + args.margem
    expandidos = expandir_obstaculos(obstaculos, raio, limites)
    limites_c = encolher_limites(limites, raio)
    if raio > 0:
        print(f"[OK] Obstaculos inflados por {raio} m "
              f"(robo {args.raio_robo} m + folga {args.margem} m): "
              f"{len(obstaculos)} -> {len(expandidos)} poligonos")

    for nome, ponto in (("inicio", args.inicio), ("objetivo", args.objetivo)):
        if not limites_mesa.ponto_dentro(ponto, limites_c):
            print(f"[ERRO] O {nome} {ponto} esta fora do limite da mesa "
                  f"calibrada (fita preta, ja descontado o raio do robo): "
                  f"escolha outro ponto (ou reduza --raio-robo).")
            return
        if any(dist_ponto_poligono(np.array(ponto), o) <= 0
               for o in expandidos):
            print(f"[ERRO] O {nome} {ponto} esta dentro de um obstaculo: "
                  f"escolha outro ponto (ou reduza --raio-robo).")
            return

    if args.max_pontos and args.max_pontos < 2:
        print("[ERRO] --max-pontos precisa ser 0 (sem limite) ou >= 2: "
              "a trajetoria tem no minimo o inicio e o objetivo")
        return
    if args.max_pontos and args.min_pontos > args.max_pontos:
        print(f"[ERRO] --min-pontos ({args.min_pontos}) nao pode passar de "
              f"--max-pontos ({args.max_pontos})")
        return

    if args.metodo == "grid":
        caminho, sucesso = plan_grid(
            args.inicio, args.objetivo, expandidos, limites_c,
            passo=args.passo, suavizar=not args.sem_suavizar,
            min_pontos=args.min_pontos, max_pontos=args.max_pontos)
    else:
        caminho, sucesso = plan_potencial(
            args.inicio, args.objetivo, expandidos,
            alpha=args.alpha, max_iter=args.max_iter,
            k_attr=args.k_attr, k_rep=args.k_rep, d0=args.d0,
            limites=limites_c)
        # A descida de gradiente devolve um ponto por passo de
        # integracao (milhares): aqui ela vira uma poligonal curta,
        # testando cada atalho contra os obstaculos.
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
        print(f"[OK] Objetivo alcancado: {len(caminho)} waypoints, "
              f"comprimento {comprimento:.4f} m")
    elif args.metodo == "grid":
        print("[AVISO] Nenhum caminho encontrado")
    else:
        print(f"[AVISO] Parou em {caminho[-1].round(4).tolist()} "
              f"apos {len(caminho)} passos (possivel minimo local)")

    # Orientacao de cada waypoint: direcao do segmento que sai dele.
    thetas = orientacoes(caminho)
    theta_inicial = robo.get("theta_rad") if robo else None

    with open(args.saida, "w", encoding="utf-8") as f:
        json.dump({
            "inicio_xy_m": list(args.inicio),
            "objetivo_xy_m": list(args.objetivo),
            "theta_inicial_robo_rad": theta_inicial,
            "metodo": args.metodo,
            "passo_grid_m": args.passo if args.metodo == "grid" else None,
            "sucesso": bool(sucesso),
            "num_pontos": int(len(caminho)),
            "max_pontos": int(args.max_pontos) if args.max_pontos else None,
            "raio_robo_m": args.raio_robo,
            "margem_seguranca_m": args.margem,
            "raio_inflado_m": raio,
            "trajetoria_xy_m": caminho.round(5).tolist(),
            "theta_rad": thetas.round(5).tolist(),
            "theta_graus": np.degrees(thetas).round(2).tolist(),
            "trajetoria_xytheta": [
                [round(x, 5), round(y, 5), round(t, 5)]
                for (x, y), t in zip(caminho, thetas)
            ],
        }, f, indent=4)
    print(f"[OK] Trajetoria salva em '{args.saida}'")

    # A grade e refeita so para a figura; o custo e desprezivel.
    grid = None
    if args.metodo == "grid":
        livre, origem = construir_grid(expandidos, limites_c, args.passo)
        objetivo_ij = _celula(args.objetivo, origem, args.passo, livre.shape)
        objetivo_ij = _celula_livre_mais_proxima(objetivo_ij, livre)
        valores = (mapa_wavefront(livre, objetivo_ij)
                   if objetivo_ij is not None else None)
        grid = (livre, origem, args.passo, valores)

    try:
        # Numa grade grossa a figura e a tabela de valores; numa grade
        # fina os numeros nao caberiam e o campo vai em cores.
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
        print("[AVISO] matplotlib indisponivel; figura nao gerada")

    
    if sucesso and ser is not None:
        # O primeiro ponto da trajetoria e a pose atual do robo, entao
        # nao vira waypoint. Para cada waypoint caminho[k+1], o angulo
        # da reta que chega nele (caminho[k] -> caminho[k+1]) e thetas[k],
        # isto e, thetas[:-1] (evitando desalinhamento entre a reta percorrida
        # e o angulo th enviado ao firmware nos pontos intermediarios).
        waypoints = caminho[1:].copy()
        thetas_deg = np.degrees(thetas)[:-1]
        theta_inicial_deg = robo.get("theta_graus") if robo else None
        if theta_inicial_deg is None and robo and "theta_rad" in robo:
            theta_inicial_deg = float(np.degrees(robo["theta_rad"]))
        if theta_inicial_deg is None:
            theta_inicial_deg = float(np.degrees(thetas[0]))

        rx = float(robo.get("x_m")) if robo else float(caminho[0, 0])
        ry = float(robo.get("y_m")) if robo else float(caminho[0, 1])
        pos_ini = np.array([rx, ry], dtype=float)

        # Aplica fator de correção de escala de odometria (caso o robô ande sistematicamente mais que o comandado)
        if abs(args.fator_odometria - 1.0) > 1e-6:
            waypoints = pos_ini + args.fator_odometria * (waypoints - pos_ini)
            print(f"[INFO] Fator de odometria {args.fator_odometria:.3f} aplicado aos waypoints.")

        if args.malha_fechada and args.cam is not None:
            print(f"[OK] Modo MALHA FECHADA ativado com câmera '{args.cam}'.")
            cur_x, cur_y, cur_th = rx, ry, theta_inicial_deg
            for idx_wp, (setpoint, th_d) in enumerate(zip(waypoints, thetas_deg)):
                print(f"\n--- Etapa {idx_wp + 1}/{len(waypoints)}: indo para {setpoint.round(4).tolist()} ---")
                stop(ser)
                set_pose(ser, cur_x, cur_y, cur_th)
                print(f"[INFO] Pose atual no robô: ({cur_x:.3f}, {cur_y:.3f}, {cur_th:.1f}°)")
                add_waypoint(ser, setpoint[0], setpoint[1], th_d)
                ok_st, resp_st, msg_st = start(ser)
                print(f"[INFO] Navegando: {resp_st} ({msg_st})")
                wait_until_done_telemetry(ser)
                time.sleep(0.4)
                novo_robo = detectar_pose_camera(args.cam)
                if novo_robo is not None:
                    cur_x = float(novo_robo["x_m"])
                    cur_y = float(novo_robo["y_m"])
                    cur_th = float(novo_robo["theta_graus"])
                    erro_wp = float(np.hypot(setpoint[0] - cur_x, setpoint[1] - cur_y))
                    print(f"[CÂMERA] Pose real medida: ({cur_x:.3f}m, {cur_y:.3f}m, {cur_th:.1f}°) | Erro ao alvo: {erro_wp:.3f}m")
            # Correção final se ainda estiver a mais de 5 cm do objetivo
            obj_x, obj_y = float(args.objetivo[0]), float(args.objetivo[1])
            erro_final = float(np.hypot(obj_x - cur_x, obj_y - cur_y))
            if erro_final > 0.05:
                print(f"\n--- Ajuste fino final para o objetivo ({obj_x:.3f}, {obj_y:.3f}) (erro={erro_final:.3f}m) ---")
                stop(ser)
                set_pose(ser, cur_x, cur_y, cur_th)
                th_corr = float(np.degrees(np.arctan2(obj_y - cur_y, obj_x - cur_x)))
                wp_corr = np.array([cur_x, cur_y]) + args.fator_odometria * np.array([obj_x - cur_x, obj_y - cur_y])
                add_waypoint(ser, wp_corr[0], wp_corr[1], th_corr)
                start(ser)
                wait_until_done_telemetry(ser)
        else:
            print(f"[OK] Enviando {len(waypoints)} waypoints "
                  f"({len(caminho)} pontos de trajetoria, limite "
                  f"{args.max_pontos or 'nenhum'})")

            # Limpa a lista anterior de waypoints no firmware (<C>) antes de definir a pose e adicionar os novos pontos
            stop(ser)

            # O firmware do robô espera o ângulo de orientação em GRAUS (e não em radianos)
            set_pose(ser, rx, ry, theta_inicial_deg)
            print(f"[INFO] Pose definida em ({rx:.3f}, {ry:.3f}, {theta_inicial_deg:.1f}°)")

            for setpoint, th_d in zip(waypoints, thetas_deg):
                add_waypoint(ser, setpoint[0], setpoint[1], th_d)
                print(f"[INFO] Waypoint {setpoint.round(4).tolist()} (th={th_d:.1f}°) adicionado!")
            ok_st, resp_st, msg_st = start(ser)
            print(f"[INFO] Início da navegação: {resp_st} ({msg_st})")
            if ok_st and not args.sem_telemetria:
                print("[INFO] Monitorando telemetria <T> do robô até concluir (Ctrl+C para interromper)...")
                try:
                    done, last_t = wait_until_done_telemetry(ser)
                    if done and last_t is not None:
                        print(f"[OK] Robô concluiu navegação! Pose final pela odometria interna: "
                              f"({last_t['x']:.3f}m, {last_t['y']:.3f}m, {last_t['th_deg']:.1f}°)")
                except KeyboardInterrupt:
                    print("\n[AVISO] Monitoramento interrompido pelo usuário. Enviando <C> (stop)...")
                    stop(ser)

if __name__ == "__main__":
    main()
