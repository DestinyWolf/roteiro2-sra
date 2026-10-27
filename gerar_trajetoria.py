"""Geracao de trajetoria por campos potenciais.

Le o mapa de obstaculos poligonais produzido por 'gerar_mapa.py'
(mapa_obstaculos.json) e gera uma trajetoria do ponto inicial ate o
objetivo por descida de gradiente sobre o potencial total
U = U_atrativo + U_repulsivo.
"""

import argparse
import json

import numpy as np


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

def expandir_obstaculos(obstaculos, raio, limites, passo=0.002):
    """Soma de Minkowski de cada obstaculo com um disco de raio 'raio'.

    Implementada como dilatacao morfologica com elemento estruturante
    circular: por definicao, a dilatacao de um conjunto por um disco E a
    soma de Minkowski dos dois. Fazer isso sobre uma mascara rasterizada
    (e nao deslocando as arestas uma a uma) resolve de graca os dois
    casos em que o offset analitico falha: vertices reflexos, onde as
    arestas deslocadas se auto-interceptam, e obstaculos que se fundem
    ao crescer.

    O resultado e o obstaculo no espaco de configuracao (C-space): com
    ele o robo de raio 'raio' pode ser tratado como uma particula.

    Pode devolver menos poligonos que a entrada, quando dois obstaculos
    se encostam apos a expansao.
    """
    if raio <= 0.0:
        return [np.asarray(o, dtype=float) for o in obstaculos]

    import cv2

    x0, y0 = limites["x_min"], limites["y_min"]
    margem = raio + 4 * passo
    largura = int(np.ceil((limites["x_max"] - x0 + 2 * margem) / passo))
    altura = int(np.ceil((limites["y_max"] - y0 + 2 * margem) / passo))

    def para_pixel(pts):
        return np.round((np.asarray(pts) - [x0 - margem, y0 - margem])
                        / passo).astype(np.int32)

    mascara = np.zeros((altura, largura), dtype=np.uint8)
    for poligono in obstaculos:
        cv2.fillPoly(mascara, [para_pixel(poligono)], 255)

    raio_px = int(np.ceil(raio / passo))
    disco = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * raio_px + 1, 2 * raio_px + 1))
    mascara = cv2.dilate(mascara, disco)

    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    expandidos = []
    for c in contornos:
        # Simplifica mantendo o erro bem abaixo da resolucao do raster.
        c = cv2.approxPolyDP(c, 0.8 * passo / passo * 1.0, True)
        pts = c.reshape(-1, 2).astype(float) * passo + [x0 - margem,
                                                        y0 - margem]
        if len(pts) >= 3:
            expandidos.append(pts)
    return expandidos


def encolher_limites(limites, raio):
    """Workspace reduzido de 'raio': o C-space das paredes."""
    if not limites or raio <= 0.0:
        return limites
    return {
        "x_min": limites["x_min"] + raio,
        "x_max": limites["x_max"] - raio,
        "y_min": limites["y_min"] + raio,
        "y_max": limites["y_max"] - raio,
    }


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
        return np.array([
            np.clip(p[0], limites["x_min"], limites["x_max"]),
            np.clip(p[1], limites["y_min"], limites["y_max"]),
        ])

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


# ------------------------------------------------------------------- mapa

def carregar_mapa(caminho_json):
    """Le obstaculos e limites do workspace do JSON de 'gerar_mapa.py'."""
    with open(caminho_json, "r", encoding="utf-8") as f:
        dados = json.load(f)

    obstaculos = [
        np.array(obs["vertices_xy_m"], dtype=float)
        for obs in dados.get("obstaculos", [])
    ]
    limites = dados.get("workspace_limites_m", {})
    return obstaculos, limites


def mapa_exemplo():
    """Mapa sintetico usado quando nao ha JSON disponivel."""
    return [
        poligono_regular((0.3, 0.3), 4, 0.08),
        poligono_regular((0.7, 0.5), 3, 0.10),
        poligono_regular((0.5, 0.65), 6, 0.09),
    ], {"x_min": 0.0, "x_max": 1.2, "y_min": 0.0, "y_max": 0.8}


def plotar(caminho, obstaculos, inicio, objetivo, limites, arquivo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(8, 6))
    for poligono in obstaculos:
        ax.add_patch(Polygon(poligono, closed=True, facecolor="0.6",
                             edgecolor="k"))
    ax.plot(caminho[:, 0], caminho[:, 1], "-b", lw=2, label="trajetoria")
    ax.plot(*inicio, "go", ms=10, label="inicio")
    ax.plot(*objetivo, "r*", ms=15, label="objetivo")
    ax.set_xlim(limites.get("x_min", 0.0), limites.get("x_max", 1.2))
    ax.set_ylim(limites.get("y_min", 0.0), limites.get("y_max", 0.8))
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(arquivo, dpi=120, bbox_inches="tight")
    print(f"[OK] Figura salva em '{arquivo}'")


def main():
    ap = argparse.ArgumentParser(
        description="Gera trajetoria por campos potenciais.")
    ap.add_argument("--mapa", default="mapa_obstaculos.json")
    ap.add_argument("--inicio", type=float, nargs=2, default=[0.05, 0.05])
    ap.add_argument("--objetivo", type=float, nargs=2, default=[1.1, 0.7])
    ap.add_argument("--k-attr", type=float, default=1.0)
    ap.add_argument("--k-rep", type=float, default=0.001)
    ap.add_argument("--d0", type=float, default=0.08)
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--saida", default="trajetoria.json")
    ap.add_argument("--figura", default="trajetoria.png")
    args = ap.parse_args()

    try:
        obstaculos, limites = carregar_mapa(args.mapa)
        print(f"[OK] {len(obstaculos)} obstaculos lidos de '{args.mapa}'")
    except FileNotFoundError:
        obstaculos, limites = mapa_exemplo()
        print(f"[AVISO] '{args.mapa}' nao encontrado; usando mapa de exemplo")

    caminho, sucesso = plan_potencial(
        args.inicio, args.objetivo, obstaculos,
        alpha=args.alpha, max_iter=args.max_iter,
        k_attr=args.k_attr, k_rep=args.k_rep, d0=args.d0, limites=limites)

    if sucesso:
        print(f"[OK] Objetivo alcancado em {len(caminho)} passos")
    else:
        print(f"[AVISO] Parou em {caminho[-1].round(4).tolist()} "
              f"apos {len(caminho)} passos (possivel minimo local)")

    with open(args.saida, "w", encoding="utf-8") as f:
        json.dump({
            "inicio_xy_m": list(args.inicio),
            "objetivo_xy_m": list(args.objetivo),
            "sucesso": bool(sucesso),
            "num_pontos": int(len(caminho)),
            "trajetoria_xy_m": caminho.round(5).tolist(),
        }, f, indent=4)
    print(f"[OK] Trajetoria salva em '{args.saida}'")

    try:
        plotar(caminho, obstaculos, args.inicio, args.objetivo, limites,
               args.figura)
    except ImportError:
        print("[AVISO] matplotlib indisponivel; figura nao gerada")


if __name__ == "__main__":
    main()
