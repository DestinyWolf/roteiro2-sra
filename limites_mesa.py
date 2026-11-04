"""Contorno real da mesa (interior da fita preta) vindo da calibracao.

'gerar_mapa.py' exporta em 'mapa_obstaculos.json' o bloco
'workspace_contorno_xy_m': o contorno em formato de estadio da mesa, em
metros. Os planejadores de trajetoria usam esse contorno como limite do
workspace (alem do retangulo x_min..x_max / y_min..y_max, que so e o
envolvente): o robo nao pode sair da regiao delimitada pela fita.

O contorno viaja dentro do dicionario 'limites', na chave "contorno"
(array N x 2), e e encolhido junto com ele por 'encolher_limites'.
"""
import numpy as np


def contorno_de_limites(limites):
    """Contorno (N x 2) guardado em 'limites', ou None."""
    if not limites:
        return None
    c = limites.get("contorno")
    return None if c is None else np.asarray(c, dtype=float)


def dentro_varios(pontos, poligono):
    """Ray casting vetorizado: mascara dos pontos (N x 2) dentro do poligono."""
    pontos = np.asarray(pontos, dtype=float).reshape(-1, 2)
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


def ponto_dentro(q, limites):
    """Verdadeiro se q esta dentro do retangulo e do contorno de 'limites'."""
    if not limites:
        return True
    q = np.asarray(q, dtype=float)
    if "x_min" in limites and not (
            limites["x_min"] <= q[0] <= limites["x_max"]
            and limites["y_min"] <= q[1] <= limites["y_max"]):
        return False
    c = contorno_de_limites(limites)
    return True if c is None else bool(dentro_varios(q, c)[0])


def amostras_dentro(amostras, limites):
    """Mascara dos pontos (N x 2) dentro do contorno (True se nao houver)."""
    c = contorno_de_limites(limites)
    if c is None:
        return np.ones(len(amostras), dtype=bool)
    return dentro_varios(amostras, c)


def ponto_mais_proximo_contorno(q, contorno):
    """Ponto da borda do contorno mais proximo de q."""
    q = np.asarray(q, dtype=float)
    melhor, d_melhor = None, np.inf
    n = len(contorno)
    for i in range(n):
        a, b = contorno[i], contorno[(i + 1) % n]
        ab = b - a
        den = float(ab @ ab)
        t = 0.0 if den == 0.0 else float(np.clip((q - a) @ ab / den, 0.0, 1.0))
        p = a + t * ab
        d = float(np.linalg.norm(q - p))
        if d < d_melhor:
            melhor, d_melhor = p, d
    return melhor


def encolher_contorno(contorno, margem, passo=0.002):
    """Afasta o contorno 'margem' metros para dentro (erosao com disco).

    Mantem o robo (de raio 'margem') totalmente dentro da fita. Devolve o
    novo contorno (N x 2), ou o original se a margem for zero.
    """
    contorno = np.asarray(contorno, dtype=float)
    if margem <= 0.0:
        return contorno

    import cv2

    origem = contorno.min(axis=0) - 4 * passo
    tam = np.ceil((contorno.max(axis=0) + 4 * passo - origem) / passo).astype(int)
    mascara = np.zeros((tam[1], tam[0]), dtype=np.uint8)
    pts = np.round((contorno - origem) / passo).astype(np.int32)
    cv2.fillPoly(mascara, [pts], 255)

    raio_px = int(np.ceil(margem / passo))
    disco = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * raio_px + 1, 2 * raio_px + 1))
    mascara = cv2.erode(mascara, disco)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contornos:
        raise ValueError(f"A margem de {margem} m nao deixa area livre dentro da mesa")
    c = max(contornos, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 0.8, True)
    return c.reshape(-1, 2).astype(float) * passo + origem

