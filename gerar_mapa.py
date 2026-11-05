import cv2
import numpy as np
import json
import glob
import re
import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import argparse

# Tabuleiro de xadrez usado na calibração: 10 x 7 QUADRADOS, cada um com 2,5 cm de lado
# (logo 9 x 6 cantos internos).
TABULEIRO_QUADRADOS = (10, 7)
TABULEIRO_LADO_M = 0.025

# Dimensões físicas reais da mesa delimitada pela fita preta (em metros)
LARGURA_MESA_PADRAO_M = 2.50
ALTURA_MESA_PADRAO_M = 1.21


def calcular_escala_cantos_px(
    cantos_px: np.ndarray,
    largura_mesa_m: float,
    altura_mesa_m: float
) -> Dict[str, float]:
    """
    A partir dos 4 cantos do retângulo envolvente da mesa na imagem
    [Topo-Esquerda, Topo-Direita, Base-Direita, Base-Esquerda] e das
    dimensões físicas da mesa (em metros), calcula quanto 1 pixel vale
    em metros (e quantos pixels equivalem a 1 metro).
    """
    c = np.asarray(cantos_px, dtype=np.float64).reshape(4, 2)
    larg_topo_px = float(np.linalg.norm(c[1] - c[0]))
    larg_base_px = float(np.linalg.norm(c[2] - c[3]))
    largura_px = 0.5 * (larg_topo_px + larg_base_px)

    alt_esq_px = float(np.linalg.norm(c[3] - c[0]))
    alt_dir_px = float(np.linalg.norm(c[2] - c[1]))
    altura_px = 0.5 * (alt_esq_px + alt_dir_px)

    ppm_x = largura_px / largura_mesa_m if largura_mesa_m > 0 else 600.0
    ppm_y = altura_px / altura_mesa_m if altura_mesa_m > 0 else 600.0
    ppm_medio = 0.5 * (ppm_x + ppm_y)

    mpp_x = largura_mesa_m / largura_px if largura_px > 0 else (1.0 / 600.0)
    mpp_y = altura_mesa_m / altura_px if altura_px > 0 else (1.0 / 600.0)
    mpp_medio = 1.0 / ppm_medio if ppm_medio > 0 else (1.0 / 600.0)

    return {
        "largura_mesa_px": round(largura_px, 3),
        "altura_mesa_px": round(altura_px, 3),
        "pixels_por_metro": round(ppm_medio, 4),
        "pixels_por_metro_x": round(ppm_x, 4),
        "pixels_por_metro_y": round(ppm_y, 4),
        "metros_por_pixel": round(mpp_medio, 6),
        "metros_por_pixel_x": round(mpp_x, 6),
        "metros_por_pixel_y": round(mpp_y, 6),
    }


# ==============================================================================
# 1. CONFIGURAÇÕES DO LABORATÓRIO E PARÂMETROS DA MESA
# ==============================================================================
@dataclass
class ConfigMapa:
    # Dimensões reais da mesa no sistema métrico (2.50 m x 1.21 m por padrão)
    largura_mesa_m: float = LARGURA_MESA_PADRAO_M   # Eixo X em metros
    altura_mesa_m: float = ALTURA_MESA_PADRAO_M    # Eixo Y em metros
    largura_camera_px: int = 1280
    altura_camera_px: int = 720

    # Escala métrica da imagem retificada (definida automaticamente pela calibração:
    # quantos pixels equivalem a 1 metro e quanto 1 pixel vale em metros)
    pixels_por_metro: float = 480.84
    metros_por_pixel: float = 1.0 / 480.84
    
    # Modo de detecção da mesa:
    # True  -> Tenta achar o maior quadrilátero automaticamente (cai para manual se falhar)
    # False -> Abre a imagem para você clicar nos 4 cantos da mesa (ignora o chão com 100% de certeza)
    auto_detectar_mesa: bool = False
    
    # Caso a câmera esteja fixa em um suporte, você pode salvar os 4 pontos (em pixels da foto original)
    # aqui após a primeira calibração para não precisar clicar toda vez. Deixe None para desativar.
    pontos_mesa_fixos: List[Tuple[int, int]] = None
    
    # Origem do sistema de coordenadas (x, y) do robô:
    # True  -> Origem (0,0) no canto INFERIOR esquerdo (Padrão Cartesiano / ROS / Robótica)
    # False -> Origem (0,0) no canto SUPERIOR esquerdo (Padrão de Imagem)
    origem_cartesiana_inferior: bool = False  # False = origem no canto superior esquerdo (eixo Y para baixo); True = origem inferior esquerda

    # Parâmetros de detecção dos objetos na mesa
    margem_borda_m: float = 0.02       # Ignora ruídos a menos de 2 cm da borda da mesa
    area_min_objeto_m2: float = 0.0009 # Área mínima do objeto (ex: 3cm x 3cm = 0.0009 m²)
    epsilon_poligono: float = 0.025    # Fator de simplificação dos vértices (0.015 a 0.04)
    inverter_binarizacao: bool = False # True = objetos escuros em mesa clara | False = objetos claros em mesa escura

    # Parâmetros para a detecção do robô (Vermelho / Laranja em HSV)
    hsv_h_min1: int = 0
    hsv_h_max1: int = 24
    hsv_h_min2: int = 165
    hsv_h_max2: int = 179
    hsv_s_min: int = 110
    hsv_v_min: int = 80
    area_min_robo_m2: float = 0.0004   # Área mínima do marcador (ex: 2cm x 2cm = 0.0004 m²)
    margem_exclusao_robo: float = 0.18 # Raio extra (em metros) para apagar o robô da máscara de obstáculos

    # Orientação do robô: o marcador vermelho fica na parte traseira/centro;
    # a roda castor fica na FRENTE, na ponta da haste triangular.
    # O algoritmo localiza a roda castor e calcula o vetor de direção do robô.
    raio_busca_robo_m: float = 0.25    # Distância máxima do marcador em que se procura o corpo/castor do robô
    limiar_corpo_robo_v: int = 90      # V (HSV) abaixo disso = parte escura do robô (rodas, chassi, hastes, castor)
    offset_theta_robo_graus: float = 0.0  # Correção angular adicional (caso haja algum desalinhamento mecânico)
    offset_centro_robo_m: float = 0.075   # Distância (m) do marcador vermelho traseiro até o centro do eixo das rodas motrizes

    # --- Calibração métrica (tabuleiro) + mesa delimitada por fita preta ---
    # Preenchidos automaticamente ao carregar o arquivo de calibração (não edite à mão)
    cantos_calibrados: Optional[List[List[float]]] = None  # 4 cantos (virtuais) do retângulo envolvente da mesa, em px da foto
    contorno_mesa_px: Optional[List[List[float]]] = None   # Contorno interno da fita preta, em px da foto original

    # Parâmetros da detecção da fita preta
    limiar_fita_v: int = 60            # Pixels com V (HSV) abaixo disso são considerados "fita preta"
    fechamento_fita_m: float = 0.01    # Fecha falhas/reflexos na fita de até ~1 cm
    area_min_mesa_m2: float = 0.10     # Área mínima aceitável para o interior da mesa
    limite_busca_mesa_m: float = 2.5   # Distância máxima (a partir do tabuleiro) em que a mesa é procurada
    max_lado_canvas_px: int = 3000     # Limite de resolução da imagem usada na busca da fita
    sem_gui: bool = False              # True = não abre janelas; salva as imagens de conferência em .png


# ==============================================================================
# 2. FUNÇÕES DE DETECÇÃO E SELEÇÃO DA MESA
# ==============================================================================
def ordenar_4_pontos(pts: np.ndarray) -> np.ndarray:
    """
    Ordena 4 pontos arbitrariamente clicados ou detectados na ordem:
    [Topo-Esquerda, Topo-Direita, Base-Direita, Base-Esquerda]
    """
    rect = np.zeros((4, 2), dtype="float32")
    soma = pts.sum(axis=1)
    rect[0] = pts[np.argmin(soma)] # Menor soma (x+y) -> Topo-Esquerda
    rect[2] = pts[np.argmax(soma)] # Maior soma (x+y) -> Base-Direita

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Menor diferença (y-x) -> Topo-Direita
    rect[3] = pts[np.argmax(diff)] # Maior diferença (y-x) -> Base-Esquerda
    return rect


def selecionar_mesa_manual(imagem: np.ndarray) -> np.ndarray:
    """
    Abre uma janela interativa para o usuário clicar nos 4 cantos da mesa.
    """
    pontos = []
    img_copia = imagem.copy()
    nome_janela = "Calibracao: Clique nos 4 cantos da mesa (ESC para sair)"

    def callback_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pontos) < 4:
            pontos.append((x, y))
            cv2.circle(img_copia, (x, y), 6, (0, 0, 255), -1)
            if len(pontos) > 1:
                cv2.line(img_copia, pontos[-2], pontos[-1], (0, 255, 0), 2)
            if len(pontos) == 4:
                cv2.line(img_copia, pontos[3], pontos[0], (0, 255, 0), 2)
            cv2.imshow(nome_janela, img_copia)

    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
    cv2.imshow(nome_janela, img_copia)
    cv2.setMouseCallback(nome_janela, callback_mouse)

    print("\n[INSTRUÇÃO] Clique nos 4 cantos da mesa na janela aberta.")
    print("Após clicar no 4º canto, pressione qualquer tecla para continuar...")

    while len(pontos) < 4:
        if cv2.waitKey(20) & 0xFF == 27:
            raise RuntimeError("Calibração cancelada pelo usuário.")

    cv2.waitKey(0)
    cv2.destroyWindow(nome_janela)

    print(f"[DICA] Se a câmera estiver fixa, copie estes pontos para 'pontos_mesa_fixos': {pontos}")
    return ordenar_4_pontos(np.array(pontos, dtype="float32"))


def detectar_mesa_auto(imagem: np.ndarray) -> np.ndarray:
    """
    Tenta localizar automaticamente o contorno de 4 lados da mesa na imagem.
    """
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(cinza, (7, 7), 0)
    bordas = cv2.Canny(blur, 50, 150)
    
    # Dilata para fechar possíveis falhas no contorno da borda da mesa
    kernel = np.ones((5, 5), np.uint8)
    bordas = cv2.dilate(bordas, kernel, iterations=2)

    contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

    area_total_img = imagem.shape[0] * imagem.shape[1]

    for c in contornos[:5]:
        area = cv2.contourArea(c)
        if area < 0.15 * area_total_img: # A mesa deve ocupar pelo menos 15% da foto
            continue
        perimetro = cv2.arcLength(c, True)
        aprox = cv2.approxPolyDP(c, 0.02 * perimetro, True)
        if len(aprox) == 4:
            print("[INFO] Mesa detectada automaticamente com sucesso!")
            return ordenar_4_pontos(aprox.reshape(4, 2).astype("float32"))

    print("[AVISO] Detecção automática não encontrou 4 cantos claros. Mudando para seleção manual...")
    return selecionar_mesa_manual(imagem)


# ==============================================================================
# 2B. CALIBRAÇÃO MÉTRICA COM TABULEIRO + DETECÇÃO DA MESA PELA FITA PRETA
# ==============================================================================
# Ideia geral:
#   1. O tabuleiro (de quadrados com lado conhecido) está sobre o tampo da mesa.
#      Com os cantos internos detectados, calcula-se uma homografia
#      "pixel da foto -> metros no plano da mesa". Como TODO o tampo está no
#      mesmo plano, essa homografia vale para a mesa inteira, mesmo que o
#      tabuleiro ocupe só uma parte dela.
#   2. A foto é retificada para uma vista de topo métrica e a fita preta é
#      segmentada. O interior da fita (região clara fechada que contém o
#      tabuleiro) é o espaço de trabalho -> contorno em formato de "estádio".
#   3. Calcula-se o retângulo envolvente mínimo do contorno: ele define o
#      sistema de coordenadas (X = lado comprido) e as dimensões da mesa.
#      Os 4 cantos desse retângulo (projetados de volta na foto) são usados
#      pela retificação normal do programa.
# IMPORTANTE: a câmera NÃO pode se mover entre a calibração e o mapeamento.

def aplicar_homografia(pontos, H: np.ndarray) -> np.ndarray:
    """Aplica a homografia H a uma lista de pontos (N x 2)."""
    pts = np.asarray(pontos, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def detectar_tabuleiro(imagem: np.ndarray, quadrados_x: int, quadrados_y: int):
    """
    Detecta os cantos internos do tabuleiro quadriculado.
    quadrados_x / quadrados_y = número de QUADRADOS em cada direção
    (o número de cantos internos é quadrados - 1).
    Retorna (padrao_cantos_internos, cantos_px (N x 2)).
    """
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    tentativas = [(quadrados_x - 1, quadrados_y - 1), (quadrados_y - 1, quadrados_x - 1)]

    # Em fotos de baixa resolução o tabuleiro ocupa poucos pixels e o detector
    # falha na escala original: tenta também com a imagem ampliada.
    for escala in (1, 2, 3, 4, 6):
        if escala == 1:
            img_busca = cinza
        else:
            img_busca = cv2.resize(cinza, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)

        for padrao in tentativas:
            ok, cantos = False, None
            if hasattr(cv2, "findChessboardCornersSB"):
                ok, cantos = cv2.findChessboardCornersSB(
                    img_busca, padrao, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
                )
            if not ok:
                ok, cantos = cv2.findChessboardCorners(
                    img_busca, padrao, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                )
                if ok:
                    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
                    cantos = cv2.cornerSubPix(img_busca, cantos, (11, 11), (-1, -1), criterio)
            if ok:
                cantos = cantos.reshape(-1, 2).astype(np.float64)
                if escala != 1:
                    cantos = (cantos + 0.5) / escala - 0.5   # volta para os pixels da foto original
                    print(f"[INFO] Tabuleiro detectado com a imagem ampliada {escala}x (foto de baixa resolução).")
                return padrao, cantos.astype(np.float32)

    raise RuntimeError(
        f"Tabuleiro de {quadrados_x}x{quadrados_y} quadrados não encontrado na imagem. "
        "Confira a contagem de QUADRADOS, a iluminação (sem reflexos) e se ele está inteiro na foto."
    )


def detectar_interior_fita(
    vista_topo: np.ndarray,
    mascara_valida: np.ndarray,
    ponto_interno_px: Optional[Tuple[float, float]],
    ppm: float,
    cfg: ConfigMapa
) -> Optional[np.ndarray]:
    """
    Na vista de topo métrica, segmenta a fita preta e devolve o contorno da
    região clara delimitada por ela (interior da mesa), em pixels da vista de topo.
    """
    alt, larg = vista_topo.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(vista_topo, (5, 5), 0), cv2.COLOR_BGR2HSV)

    # "Escuro" = fita preta (ou região fora da foto)
    escuro = np.where(hsv[:, :, 2] < cfg.limiar_fita_v, 255, 0).astype(np.uint8)
    escuro[mascara_valida == 0] = 255

    # Fecha pequenas falhas na fita (reflexos, emendas)
    k = max(3, int(round(cfg.fechamento_fita_m * ppm)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    escuro = cv2.morphologyEx(escuro, cv2.MORPH_CLOSE, kernel)

    livre = cv2.bitwise_not(escuro)
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(livre, connectivity=4)

    invalido_dil = cv2.dilate((mascara_valida == 0).astype(np.uint8), np.ones((5, 5), np.uint8))
    area_min_px = cfg.area_min_mesa_m2 * ppm ** 2
    ordem = sorted(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)

    reserva = None
    candidatos = []
    for i in ordem:
        x, y, w, h, area = stats[i]
        if area < area_min_px:
            break
        # O interior da fita é fechado: não pode encostar na borda da imagem
        if x <= 0 or y <= 0 or x + w >= larg or y + h >= alt:
            continue
        comp = (rotulos == i).astype(np.uint8)
        if np.any(comp & invalido_dil):
            continue

        contornos, _ = cv2.findContours(comp * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(contornos, key=cv2.contourArea)

        if ponto_interno_px is None:
            return c
        if cv2.pointPolygonTest(c, (float(ponto_interno_px[0]), float(ponto_interno_px[1])), False) >= 0:
            candidatos.append(c)
        elif reserva is None:
            reserva = c

    # Regiões "em anel" (ex: chão em volta da mesa) também contêm o tabuleiro quando
    # preenchidas; a mesa é a região MAIS INTERNA (menor área preenchida).
    if candidatos:
        return min(candidatos, key=cv2.contourArea)

    if reserva is not None:
        print("[AVISO] Região delimitada pela fita encontrada, mas ela não contém o tabuleiro. Confira o resultado!")
    return reserva


def calibrar_com_tabuleiro(
    imagem: np.ndarray,
    quadrados_x: int,
    quadrados_y: int,
    lado_quadrado_m: float,
    cfg: ConfigMapa
) -> Optional[Dict]:
    """
    Executa a calibração completa (escala métrica + detecção da mesa pela fita).
    Retorna o dicionário a ser salvo em JSON, ou None se o usuário cancelar.
    """
    h_img, w_img = imagem.shape[:2]

    # --- 1. Tabuleiro -> homografia foto(px) -> plano da mesa(m) ---
    padrao, cantos_px = detectar_tabuleiro(imagem, quadrados_x, quadrados_y)
    obj_m = np.array(
        [[i * lado_quadrado_m, j * lado_quadrado_m] for j in range(padrao[1]) for i in range(padrao[0])],
        dtype=np.float64
    )
    H, _ = cv2.findHomography(cantos_px.astype(np.float64), obj_m, 0)
    if H is None:
        raise RuntimeError("Falha ao calcular a homografia do tabuleiro.")
    H_inv = np.linalg.inv(H)

    erros = np.linalg.norm(aplicar_homografia(cantos_px, H) - obj_m, axis=1)
    print(f"[CALIB] Tabuleiro: {padrao[0]}x{padrao[1]} cantos internos | "
          f"erro médio = {erros.mean() * 1000:.2f} mm | máx = {erros.max() * 1000:.2f} mm")

    # --- 2. Vista de topo métrica ao redor do tabuleiro ---
    centro_tab_m = obj_m.mean(axis=0)
    cantos_foto_m = aplicar_homografia([[0, 0], [w_img - 1, 0], [w_img - 1, h_img - 1], [0, h_img - 1]], H)
    lim = cfg.limite_busca_mesa_m
    xmin = max(cantos_foto_m[:, 0].min(), centro_tab_m[0] - lim)
    xmax = min(cantos_foto_m[:, 0].max(), centro_tab_m[0] + lim)
    ymin = max(cantos_foto_m[:, 1].min(), centro_tab_m[1] - lim)
    ymax = min(cantos_foto_m[:, 1].max(), centro_tab_m[1] + lim)
    if xmin >= xmax or ymin >= ymax:
        xmin, xmax = centro_tab_m[0] - lim, centro_tab_m[0] + lim
        ymin, ymax = centro_tab_m[1] - lim, centro_tab_m[1] + lim

    ppm = min(float(cfg.pixels_por_metro), cfg.max_lado_canvas_px / max(xmax - xmin, ymax - ymin))
    A = np.array([[ppm, 0, -ppm * xmin], [0, ppm, -ppm * ymin], [0, 0, 1]], dtype=np.float64)
    tam = (int(math.ceil((xmax - xmin) * ppm)), int(math.ceil((ymax - ymin) * ppm)))

    vista_topo = cv2.warpPerspective(imagem, A @ H, tam)
    valida = cv2.warpPerspective(np.full((h_img, w_img), 255, np.uint8), A @ H, tam, flags=cv2.INTER_NEAREST)

    # --- 3. Interior da fita preta (deve conter o tabuleiro) ---
    centro_tab_px = aplicar_homografia([centro_tab_m], A)[0]
    contorno = detectar_interior_fita(vista_topo, valida, centro_tab_px, ppm, cfg)
    if contorno is None:
        cv2.namedWindow("Vista de topo (fita NAO encontrada)", cv2.WINDOW_NORMAL)
        cv2.imshow("Vista de topo (fita NAO encontrada)", vista_topo)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        raise RuntimeError(
            "Não foi possível encontrar a região fechada pela fita preta. "
            "Tente ajustar --limiar-fita (ex: 80 ou 100) e verifique se a mesa inteira aparece na foto."
        )

    # Simplifica o contorno (tolerância de ~2 mm) mantendo a forma das curvas
    contorno = cv2.approxPolyDP(contorno, 0.002 * ppm, True).reshape(-1, 2).astype(np.float64)
    contorno_m = contorno / ppm + np.array([xmin, ymin])          # metros (referencial do tabuleiro)
    contorno_foto_px = aplicar_homografia(contorno_m, H_inv)       # pixels da foto original

    # --- 4. Referencial da mesa: retângulo envolvente mínimo (X = lado comprido) ---
    rect = cv2.minAreaRect(contorno_m.astype(np.float32))
    caixa = cv2.boxPoints(rect).astype(np.float64)
    e1, e2 = caixa[1] - caixa[0], caixa[2] - caixa[1]
    u, v = (e1, e2) if np.linalg.norm(e1) >= np.linalg.norm(e2) else (e2, e1)
    comprimento_m, largura_m = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    centro = caixa.mean(axis=0)

    def direcao_na_foto(vetor):
        p = aplicar_homografia([centro, centro + vetor / np.linalg.norm(vetor) * 0.01], H_inv)
        return p[1] - p[0]

    # X da mesa apontando para a direita da foto e sem espelhamento da imagem
    du = direcao_na_foto(u)
    if du[0] < 0:
        u, du = -u, -du
    dv = direcao_na_foto(v)
    if du[0] * dv[1] - du[1] * dv[0] < 0:
        v = -v

    cantos_m = np.array([
        centro - u / 2 - v / 2,   # Topo-Esquerda
        centro + u / 2 - v / 2,   # Topo-Direita
        centro + u / 2 + v / 2,   # Base-Direita
        centro - u / 2 + v / 2,   # Base-Esquerda
    ])
    cantos_foto = aplicar_homografia(cantos_m, H_inv)

    # Escala medida apenas nos quadrados do tabuleiro (no plano da placa central)
    pts_tab_grid = cantos_px.reshape(padrao[1], padrao[0], 2)
    passo_tab_x_px = float(np.linalg.norm(np.diff(pts_tab_grid, axis=1), axis=2).mean())
    passo_tab_y_px = float(np.linalg.norm(np.diff(pts_tab_grid, axis=0), axis=2).mean())
    passo_tab_px = 0.5 * (passo_tab_x_px + passo_tab_y_px)
    ppm_tabuleiro = passo_tab_px / lado_quadrado_m if lado_quadrado_m > 0 else 600.0
    mpp_tabuleiro = lado_quadrado_m / passo_tab_px if passo_tab_px > 0 else (1.0 / 600.0)

    # Dimensões físicas reais da mesa (lado comprido = X = 2.50 m, lado curto = Y = 1.21 m):
    # O tabuleiro define a orientação/homografia da vista de topo e localiza a fita preta;
    # como a placa do tabuleiro tem espessura acima do tampo e ocupa só o centro da lente,
    # a escala final pixel<->metro no plano da mesa é ancorada nas dimensões reais da mesa.
    if cfg.largura_mesa_m > 0 and cfg.altura_mesa_m > 0:
        comprimento_final_m = float(max(cfg.largura_mesa_m, cfg.altura_mesa_m))
        largura_final_m = float(min(cfg.largura_mesa_m, cfg.altura_mesa_m))
    else:
        comprimento_final_m = comprimento_m
        largura_final_m = largura_m

    escala_mesa = calcular_escala_cantos_px(cantos_foto, comprimento_final_m, largura_final_m)
    ppm_calibrado = escala_mesa["pixels_por_metro"]
    mpp_calibrado = escala_mesa["metros_por_pixel"]

    print(f"[CALIB] Mesa bruta pelo tabuleiro: {comprimento_m * 100:.1f} cm x {largura_m * 100:.1f} cm "
          f"(1 px no tabuleiro = {mpp_tabuleiro * 1000:.3f} mm)")
    print(f"[CALIB] Mesa ajustada (dimensão real): {comprimento_final_m * 100:.1f} cm x {largura_final_m * 100:.1f} cm "
          f"| {escala_mesa['largura_mesa_px']:.1f} px x {escala_mesa['altura_mesa_px']:.1f} px")
    print(f"[CALIB] Escala final calibrada: {ppm_calibrado:.2f} px/m -> 1 pixel = {mpp_calibrado * 1000:.4f} mm "
          f"({mpp_calibrado:.6f} m/px)")

    calib = {
        "resolucao_px": [w_img, h_img],
        "tabuleiro": {
            "quadrados": [quadrados_x, quadrados_y],
            "lado_quadrado_m": lado_quadrado_m,
            "erro_medio_mm": round(float(erros.mean() * 1000), 3),
            "largura_medida_tabuleiro_m": round(comprimento_m, 4),
            "altura_medida_tabuleiro_m": round(largura_m, 4),
            "pixels_por_metro_tabuleiro": round(ppm_tabuleiro, 4),
            "metros_por_pixel_tabuleiro": round(mpp_tabuleiro, 6),
        },
        "largura_mesa_m": round(comprimento_final_m, 4),
        "altura_mesa_m": round(largura_final_m, 4),
        "pixels_por_metro": ppm_calibrado,
        "metros_por_pixel": mpp_calibrado,
        "escala_px_m": escala_mesa,
        "cantos_img_px": np.round(cantos_foto, 3).tolist(),
        "contorno_mesa_px": np.round(contorno_foto_px, 3).tolist(),
    }

    # --- 5. Conferência visual ---
    debug = imagem.copy()
    cv2.drawChessboardCorners(debug, padrao, cantos_px.reshape(-1, 1, 2), True)
    cv2.polylines(debug, [np.round(contorno_foto_px).astype(np.int32)], True, (255, 0, 255), 2)
    cv2.polylines(debug, [np.round(cantos_foto).astype(np.int32)], True, (0, 255, 255), 1)

    cfg_tmp = ConfigMapa(**{
        **cfg.__dict__,
        "largura_mesa_m": comprimento_final_m,
        "altura_mesa_m": largura_final_m,
        "pixels_por_metro": ppm_calibrado,
        "metros_por_pixel": mpp_calibrado,
    })
    cantos_f32 = cantos_foto.astype(np.float32)
    retificada = retificar_mesa(imagem, cantos_f32, cfg_tmp)
    contorno_ret = aplicar_homografia(contorno_foto_px, matriz_retificacao(cantos_f32, cfg_tmp))
    cv2.polylines(retificada, [np.round(contorno_ret).astype(np.int32)], True, (255, 0, 255), 2)
    cv2.putText(
        retificada,
        f"{comprimento_final_m * 100:.1f} x {largura_final_m * 100:.1f} cm | 1px = {mpp_calibrado * 1000:.3f} mm",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
    )

    if cfg.sem_gui:
        cv2.imwrite("calibracao_debug_fita.png", debug)
        cv2.imwrite("calibracao_debug_retificada.png", retificada)
        print("[CALIB] Imagens de conferência salvas em 'calibracao_debug_fita.png' e "
              "'calibracao_debug_retificada.png'.")
        tecla = 13
    else:
        cv2.namedWindow("Calibracao: tabuleiro e fita", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Calibracao: mesa retificada", cv2.WINDOW_NORMAL)
        cv2.imshow("Calibracao: tabuleiro e fita", debug)
        cv2.imshow("Calibracao: mesa retificada", retificada)
        print("\n[INSTRUÇÃO] Confira as janelas. Qualquer tecla = SALVAR | ESC = descartar.")
        tecla = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

    if tecla == 27:
        print("[CALIB] Calibração descartada.")
        return None
    return calib


def salvar_calibracao(calib: Dict, caminho: str):
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=4)
    print(f"[OK] Calibração salva em '{caminho}'")


def carregar_calibracao(caminho: str, cfg: ConfigMapa) -> bool:
    """Lê o arquivo de calibração e preenche a configuração. Retorna True se carregou."""
    if not os.path.exists(caminho):
        return False
    with open(caminho, "r", encoding="utf-8") as f:
        calib = json.load(f)

    cfg.largura_mesa_m = float(calib.get("largura_mesa_m", cfg.largura_mesa_m))
    cfg.altura_mesa_m = float(calib.get("altura_mesa_m", cfg.altura_mesa_m))
    cfg.cantos_calibrados = calib["cantos_img_px"]
    cfg.contorno_mesa_px = calib.get("contorno_mesa_px")
    cfg.largura_camera_px, cfg.altura_camera_px = calib.get(
        "resolucao_px", [cfg.largura_camera_px, cfg.altura_camera_px]
    )

    escala = calcular_escala_cantos_px(cfg.cantos_calibrados, cfg.largura_mesa_m, cfg.altura_mesa_m)
    cfg.pixels_por_metro = float(calib.get("pixels_por_metro", escala["pixels_por_metro"]))
    cfg.metros_por_pixel = float(calib.get("metros_por_pixel", 1.0 / cfg.pixels_por_metro))

    print(f"[INFO] Calibração carregada de '{caminho}': mesa de "
          f"{cfg.largura_mesa_m * 100:.1f} cm x {cfg.altura_mesa_m * 100:.1f} cm | "
          f"{cfg.pixels_por_metro:.2f} px/m (1 px = {cfg.metros_por_pixel * 1000:.4f} mm)")
    return True


# ==============================================================================
# 3. RETIFICAÇÃO, DETECÇÃO DO ROBÔ (HSV) E EXTRAÇÃO DOS POLÍGONOS
# ==============================================================================
def dimensoes_retificada_px(cfg: ConfigMapa) -> Tuple[int, int]:
    """Retorna (largura_px, altura_px) da imagem retificada usando a escala calibrada."""
    largura_px = max(1, int(round(cfg.largura_mesa_m * cfg.pixels_por_metro)))
    altura_px = max(1, int(round(cfg.altura_mesa_m * cfg.pixels_por_metro)))
    return largura_px, altura_px


def matriz_retificacao(cantos_mesa: np.ndarray, cfg: ConfigMapa) -> np.ndarray:
    """Matriz de homografia: pixels da foto original -> pixels da mesa retificada."""
    largura_px, altura_px = dimensoes_retificada_px(cfg)

    destino = np.array([
        [0, 0],
        [largura_px - 1, 0],
        [largura_px - 1, altura_px - 1],
        [0, altura_px - 1]
    ], dtype="float32")

    return cv2.getPerspectiveTransform(cantos_mesa.astype("float32"), destino)


def retificar_mesa(imagem: np.ndarray, cantos_mesa: np.ndarray, cfg: ConfigMapa) -> np.ndarray:
    """
    Aplica Homografia para recortar apenas a mesa e deixá-la na proporção métrica exata.
    """
    largura_px, altura_px = dimensoes_retificada_px(cfg)

    matriz_H = matriz_retificacao(cantos_mesa, cfg)
    mesa_retificada = cv2.warpPerspective(imagem, matriz_H, (largura_px, altura_px))
    return mesa_retificada


def pixel_para_metro(
    x_px: float,
    y_px: float,
    altura_px: int,
    cfg: ConfigMapa,
    largura_px: Optional[int] = None
) -> Tuple[float, float]:
    """
    Converte coordenadas de pixel da imagem retificada para metros (x, y)
    usando a relação exata metros_por_pixel da mesa calibrada.
    """
    if largura_px is None:
        largura_px, _ = dimensoes_retificada_px(cfg)
    mpp_x = cfg.largura_mesa_m / float(largura_px) if largura_px > 0 else cfg.metros_por_pixel
    mpp_y = cfg.altura_mesa_m / float(altura_px) if altura_px > 0 else cfg.metros_por_pixel

    x_m = x_px * mpp_x
    if cfg.origem_cartesiana_inferior:
        y_m = (altura_px - y_px) * mpp_y
    else:
        y_m = y_px * mpp_y
    return round(float(x_m), 4), round(float(y_m), 4)


def estimar_theta_robo(
    mesa_img: np.ndarray,
    imagem_hsv: np.ndarray,
    mascara_marcador: np.ndarray,
    centro_marcador_px: Tuple[float, float],
    cfg: ConfigMapa,
    mascara_mesa: Optional[np.ndarray] = None
) -> Tuple[Optional[float], Optional[Tuple[float, float]], Optional[np.ndarray]]:
    """
    Estima a orientação do robô a partir da sua roda castor dianteira em 2 etapas:

    1. Direção frontal base (chassi + rodas de borracha):
       O marcador vermelho fica na traseira do robô; logo à sua frente (r entre 3,5 cm
       e 13,5 cm) ficam a barra transversal preta do chassi e as duas rodas pretas
       simétricas. O centro de massa ponderado pelo contraste em relação ao tampo da
       mesa fornece a direção frontal aproximada sem ambiguidade.
    2. Localização exata da roda castor na ponta frontal:
       As duas rodas pretas laterais ficam a ~+-45° do eixo frontal e a r < 14 cm.
       Já a roda castor fica na ponta das hastes triangulares, a r entre 14 cm e 20,5 cm
       exatamente no cone frontal (+-28°). Avaliando o contraste adaptativo + bordas
       (Sobel) nesse setor frontal, localiza-se o centro exato da roda castor mesmo sob
       variações de iluminação ou reflexo metálico.

    Retorna:
      - theta_rad: Ângulo em radianos (no padrão cartesiano se cfg.origem_cartesiana_inferior)
      - castor_px: Coordenadas (x, y) da roda castor em pixels da imagem retificada (ou None)
      - mascara_corpo: Máscara binária do corpo do robô para exclusão segura dos obstáculos
    """
    alt, larg = mesa_img.shape[:2]
    cx, cy = centro_marcador_px
    ppm = float(cfg.pixels_por_metro)

    # Recorta uma janela local ao redor do marcador para acelerar o processamento
    r_max_m = max(cfg.raio_busca_robo_m, 0.27)
    r_win_px = int(math.ceil(r_max_m * ppm)) + 4
    x0 = max(0, int(math.floor(cx)) - r_win_px)
    x1 = min(larg, int(math.ceil(cx)) + r_win_px + 1)
    y0 = max(0, int(math.floor(cy)) - r_win_px)
    y1 = min(alt, int(math.ceil(cy)) + r_win_px + 1)

    V_win = imagem_hsv[y0:y1, x0:x1, 2].astype(np.float64)
    marc_win = mascara_marcador[y0:y1, x0:x1] > 0

    if mascara_mesa is not None:
        # Recua levemente a máscara da mesa para que a fita preta da borda nunca entre no cálculo
        margem_fita_px = max(3, int(round(0.015 * ppm)))
        k_fita = 2 * margem_fita_px + 1
        mesa_interna = cv2.erode(
            mascara_mesa[y0:y1, x0:x1],
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_fita, k_fita))
        ) > 0
    else:
        mesa_interna = np.ones_like(V_win, dtype=bool)

    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx - cx
    dy = yy - cy
    r_m = np.hypot(dx, dy) / ppm

    valido = mesa_interna & (~marc_win)

    # Brilho mediano do fundo da mesa ao redor do robô (anel de 20 cm a 27 cm)
    anel_bg = (r_m >= 0.20) & (r_m <= 0.27) & valido
    if np.count_nonzero(anel_bg) >= 20:
        v_bg = float(np.median(V_win[anel_bg]))
    else:
        anel_bg_fallback = (r_m >= 0.16) & (r_m <= 0.27) & valido
        v_bg = float(np.median(V_win[anel_bg_fallback])) if np.any(anel_bg_fallback) else 145.0

    # --- Etapa 1: Direção frontal aproximada pelo chassi preto + rodas (r de 3,5 cm a 13,5 cm) ---
    limiar_chassi = min(float(cfg.limiar_corpo_robo_v), v_bg - 30.0)
    escuro_chassi_mask = (V_win < limiar_chassi) & (r_m >= 0.030) & (r_m <= 0.140) & valido
    k_close = max(3, int(round(0.015 * ppm)) | 1)
    junto_chassi = cv2.morphologyEx(
        escuro_chassi_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    )
    n_cc, rotulos_cc = cv2.connectedComponents(junto_chassi, connectivity=8)
    comp_chassi = escuro_chassi_mask
    if n_cc > 1:
        # Mantém apenas componentes que começam próximos ao marcador (r <= 8,5 cm),
        # evitando que algum obstáculo externo próximo influencie a direção
        rotulos_perto = np.unique(rotulos_cc[(junto_chassi > 0) & (r_m <= 0.085)])
        rotulos_perto = rotulos_perto[rotulos_perto > 0]
        if len(rotulos_perto) > 0:
            comp_chassi = np.isin(rotulos_cc, rotulos_perto) & escuro_chassi_mask

    anel_chassi = (r_m >= 0.035) & (r_m <= 0.135) & comp_chassi
    if np.count_nonzero(anel_chassi) < 10:
        anel_chassi = (r_m >= 0.035) & (r_m <= 0.135) & valido

    pesos_chassi = (np.clip(v_bg - V_win - 25.0, 0.0, None) ** 2) * anel_chassi
    if pesos_chassi.sum() < 1e-3:
        return None, None, None

    dx_coarse = float((dx * pesos_chassi).sum() / pesos_chassi.sum())
    dy_coarse = float((dy * pesos_chassi).sum() / pesos_chassi.sum())
    if math.hypot(dx_coarse, dy_coarse) < 1e-4:
        return None, None, None
    ang_coarse = math.atan2(dy_coarse, dx_coarse)

    # --- Etapa 2: Localização precisa da roda castor no cone frontal (r de 14,2 cm a 20,2 cm, +-28°) ---
    # As rodas pretas laterais estão a +-45° e r < 14 cm, ficando 100% fora deste setor.
    gx = cv2.Sobel(V_win, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(V_win, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)

    ang_px = np.arctan2(dy, dx)
    diff_ang = np.abs((ang_px - ang_coarse + np.pi) % (2.0 * np.pi) - np.pi)
    setor_castor = (r_m >= 0.142) & (r_m <= 0.202) & (diff_ang <= math.radians(28.0)) & valido

    contraste_castor = (np.clip(v_bg - V_win - 15.0, 0.0, None) ** 1.5) + 0.35 * grad
    pesos_castor = contraste_castor * setor_castor

    if np.count_nonzero(pesos_castor > 0) >= 8 and pesos_castor.sum() > 1e-3:
        vals_ativos = pesos_castor[pesos_castor > 0]
        limiar_top = float(np.percentile(vals_ativos, 65))
        mask_top = setor_castor & (pesos_castor >= limiar_top)
        w_top = pesos_castor * mask_top
        castor_x = float((xx * w_top).sum() / w_top.sum())
        castor_y = float((yy * w_top).sum() / w_top.sum())
        castor_px = (castor_x, castor_y)
        dx_f = castor_x - cx
        dy_f = castor_y - cy
    else:
        castor_x = cx + math.cos(ang_coarse) * (0.162 * ppm)
        castor_y = cy + math.sin(ang_coarse) * (0.162 * ppm)
        castor_px = (castor_x, castor_y)
        dx_f, dy_f = dx_coarse, dy_coarse

    if math.hypot(dx_f, dy_f) < 1e-4:
        return None, None, None

    theta = math.atan2(-dy_f, dx_f) if cfg.origem_cartesiana_inferior else math.atan2(dy_f, dx_f)
    if cfg.offset_theta_robo_graus != 0.0:
        theta = math.atan2(
            math.sin(theta + math.radians(cfg.offset_theta_robo_graus)),
            math.cos(theta + math.radians(cfg.offset_theta_robo_graus))
        )

    # Máscara completa do corpo do robô para exclusão dos obstáculos
    mascara_corpo = np.zeros((alt, larg), dtype=np.uint8)
    corpo_win = ((v_bg - V_win) > 22.0) & (r_m <= 0.21) & valido
    mascara_corpo[y0:y1, x0:x1] = corpo_win.astype(np.uint8) * 255
    return theta, castor_px, mascara_corpo


def detectar_robo_hsv(
    mesa_img: np.ndarray,
    cfg: ConfigMapa,
    mascara_mesa: Optional[np.ndarray] = None
) -> Tuple[Optional[Dict], np.ndarray]:
    """
    Localiza o robô e sua orientação na imagem retificada da mesa.
    A posição (x, y) é obtida pelo marcador vermelho/laranja.
    A orientação (theta) é obtida a partir da roda castor frontal.
    Retorna:
      - dados_robo: Dicionário com (x_m, y_m, theta_rad, theta_graus, vertices_xy_m, castor_px, etc.) ou None
      - mascara_exclusao_robo: Máscara binária dilatada da região do robô para removê-lo dos obstáculos
    """
    altura_px, largura_px = mesa_img.shape[:2]
    blur = cv2.GaussianBlur(mesa_img, (5, 5), 0)
    imagem_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Faixa 1: Vermelho-alaranjado e Laranja (Hue 0 a 24)
    limite_inferior1 = np.array([cfg.hsv_h_min1, cfg.hsv_s_min, cfg.hsv_v_min], dtype=np.uint8)
    limite_superior1 = np.array([cfg.hsv_h_max1, 255, 255], dtype=np.uint8)
    mascara1 = cv2.inRange(imagem_hsv, limite_inferior1, limite_superior1)

    # Faixa 2: Vermelho profundo / Magenta (Hue 165 a 179)
    limite_inferior2 = np.array([cfg.hsv_h_min2, cfg.hsv_s_min, cfg.hsv_v_min], dtype=np.uint8)
    limite_superior2 = np.array([cfg.hsv_h_max2, 255, 255], dtype=np.uint8)
    mascara2 = cv2.inRange(imagem_hsv, limite_inferior2, limite_superior2)

    mascara_robo = cv2.bitwise_or(mascara1, mascara2)

    # Considera apenas o que está dentro da fita (ignora o chão nas pontas arredondadas)
    if mascara_mesa is not None:
        mascara_robo = cv2.bitwise_and(mascara_robo, mascara_mesa)

    # Limpeza morfológica para eliminar ruídos pontuais
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara_robo = cv2.morphologyEx(mascara_robo, cv2.MORPH_OPEN, kernel, iterations=1)
    mascara_robo = cv2.morphologyEx(mascara_robo, cv2.MORPH_CLOSE, kernel, iterations=2)

    contornos, _ = cv2.findContours(mascara_robo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mascara_exclusao = np.zeros_like(mascara_robo)

    if not contornos:
        return None, mascara_exclusao

    maior_contorno = max(contornos, key=cv2.contourArea)
    area_min_px = cfg.area_min_robo_m2 * (cfg.pixels_por_metro ** 2)

    if cv2.contourArea(maior_contorno) < area_min_px:
        return None, mascara_exclusao

    M = cv2.moments(maior_contorno)
    if M["m00"] == 0:
        return None, mascara_exclusao

    # Centroide do marcador vermelho em pixels e em metros
    cx_px = M["m10"] / M["m00"]
    cy_px = M["m01"] / M["m00"]
    marcador_x_m, marcador_y_m = pixel_para_metro(cx_px, cy_px, altura_px, cfg, largura_px)

    # Raio do marcador
    (_, _), raio_px = cv2.minEnclosingCircle(maior_contorno)
    raio_m = round(float(raio_px / cfg.pixels_por_metro), 4)

    # Orientação e detecção da roda castor dianteira
    theta_rad, castor_px, mascara_corpo = estimar_theta_robo(
        mesa_img, imagem_hsv, mascara_robo, (cx_px, cy_px), cfg, mascara_mesa
    )

    if theta_rad is None:
        print("[AVISO] Não foi possível localizar a roda castor do robô; usando o eixo do marcador.")
        mu20 = M["mu20"] / M["m00"]
        mu02 = M["mu02"] / M["m00"]
        mu11 = M["mu11"] / M["m00"]
        theta_img_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        theta_rad = -theta_img_rad if cfg.origem_cartesiana_inferior else theta_img_rad
        castor_px = None
    else:
        print(f"[INFO] Roda castor detectada na frente do robô. Orientação theta: {np.degrees(theta_rad):.1f}°")

    # Desloca a posição reportada do marcador traseiro para o centro cinemático do robô
    # (ponto médio do eixo entre as duas rodas motrizes, ~7.5 cm à frente do marcador na direção de theta)
    robo_x_m = round(float(marcador_x_m + cfg.offset_centro_robo_m * np.cos(theta_rad)), 4)
    robo_y_m = round(float(marcador_y_m + cfg.offset_centro_robo_m * np.sin(theta_rad)), 4)
    ang_img_robo = -theta_rad if cfg.origem_cartesiana_inferior else theta_rad
    eixo_cx_px = cx_px + (cfg.offset_centro_robo_m * cfg.pixels_por_metro) * np.cos(ang_img_robo)
    eixo_cy_px = cy_px + (cfg.offset_centro_robo_m * cfg.pixels_por_metro) * np.sin(ang_img_robo)

    # Vértices aproximados do marcador do robô
    perimetro = cv2.arcLength(maior_contorno, True)
    vertices_robo_px = cv2.approxPolyDP(maior_contorno, cfg.epsilon_poligono * perimetro, True).reshape(-1, 2)
    vertices_robo_m = np.array([pixel_para_metro(vx, vy, altura_px, cfg, largura_px) for vx, vy in vertices_robo_px])

    # Cria máscara de exclusão dilatada para apagar todo o robô (marcador, rodas, chassi e castor) da máscara de obstáculos
    margem_exclusao_px = int(cfg.margem_exclusao_robo * cfg.pixels_por_metro)
    cv2.circle(mascara_exclusao, (int(cx_px), int(cy_px)), int(raio_px + margem_exclusao_px), 255, -1)
    cv2.circle(mascara_exclusao, (int(eixo_cx_px), int(eixo_cy_px)), int(0.14 * cfg.pixels_por_metro), 255, -1)
    if mascara_corpo is not None:
        kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        corpo_dil = cv2.dilate(mascara_corpo, kernel_dil)
        mascara_exclusao = cv2.bitwise_or(mascara_exclusao, corpo_dil)
    if castor_px is not None:
        cv2.circle(mascara_exclusao, (int(castor_px[0]), int(castor_px[1])), int(0.06 * cfg.pixels_por_metro), 255, -1)

    dados_robo = {
        "x_m": robo_x_m,
        "y_m": robo_y_m,
        "marcador_x_m": marcador_x_m,
        "marcador_y_m": marcador_y_m,
        "offset_centro_m": cfg.offset_centro_robo_m,
        "theta_rad": round(float(theta_rad), 4),
        "theta_graus": round(float(np.degrees(theta_rad)), 2),
        "raio_marcador_m": raio_m,
        "vertices_xy_m": vertices_robo_m,
        "centro_px": (int(cx_px), int(cy_px)),
        "eixo_px": (int(round(eixo_cx_px)), int(round(eixo_cy_px))),
        "castor_px": (int(castor_px[0]), int(castor_px[1])) if castor_px else None,
        "raio_px": int(raio_px)
    }

    return dados_robo, mascara_exclusao


def extrair_obstaculos_poligonais(
    mesa_img: np.ndarray,
    cfg: ConfigMapa,
    mascara_exclusao_robo: Optional[np.ndarray] = None,
    dados_robo: Optional[Dict] = None,
    mascara_mesa: Optional[np.ndarray] = None
):
    """
    Identifica os objetos sobre a mesa, extrai seus vértices e converte para metros.
    Se 'mascara_mesa' (interior da fita preta) for informada, apenas essa região é analisada.
    Retorna:
      - lista_poligonos_m: Lista de matrizes NumPy (N x 2) com coordenadas (x, y) em metros
      - img_debug: Imagem anotada para conferência visual (obstáculos + robô)
      - mascara: Máscara binária final dos obstáculos
    """
    altura_px, largura_px = mesa_img.shape[:2]
    cinza = cv2.cvtColor(mesa_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(cinza, (5, 5), 0)

    # Limiarização de Otsu (separa fundo da mesa dos objetos automaticamente)
    tipo_thresh = cv2.THRESH_BINARY_INV if cfg.inverter_binarizacao else cv2.THRESH_BINARY
    if mascara_mesa is not None and np.count_nonzero(mascara_mesa) > 0:
        # Otsu calculado SOMENTE com os pixels dentro da fita (chão e fita não influenciam)
        valores = blur[mascara_mesa > 0].reshape(-1, 1)
        limiar_otsu, _ = cv2.threshold(valores, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mascara = cv2.threshold(blur, limiar_otsu, 255, tipo_thresh)
    else:
        _, mascara = cv2.threshold(blur, 0, 255, tipo_thresh + cv2.THRESH_OTSU)

    # Remove a região do robô para que ele NÃO seja detectado como obstáculo
    if mascara_exclusao_robo is not None:
        mascara = cv2.bitwise_and(mascara, cv2.bitwise_not(mascara_exclusao_robo))

    margem_px = int(cfg.margem_borda_m * cfg.pixels_por_metro)
    if mascara_mesa is not None:
        # Mantém só o interior da fita, recuado 'margem_borda_m' (a fita preta não vira obstáculo)
        area_util = mascara_mesa
        if margem_px > 0:
            k = 2 * margem_px + 1
            area_util = cv2.erode(mascara_mesa, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        mascara = cv2.bitwise_and(mascara, area_util)
    elif margem_px > 0:
        # Zera as bordas da imagem para evitar que rebarbas da quina da mesa virem obstáculos
        mascara[:margem_px, :] = 0
        mascara[-margem_px:, :] = 0
        mascara[:, :margem_px] = 0
        mascara[:, -margem_px:] = 0

    # Operação morfológica para fechar pequenos buracos de reflexo de luz nos objetos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area_min_px = cfg.area_min_objeto_m2 * (cfg.pixels_por_metro ** 2)
    lista_poligonos_m = []
    img_debug = mesa_img.copy()

    for idx, c in enumerate(contornos):
        area_px = cv2.contourArea(c)
        if area_px < area_min_px:
            continue

        # Aproximação poligonal (Ramer-Douglas-Peucker) para obter apenas os vértices das quinas
        perimetro = cv2.arcLength(c, True)
        vertices_px = cv2.approxPolyDP(c, cfg.epsilon_poligono * perimetro, True)
        vertices_px = vertices_px.reshape(-1, 2) # Formato (N, 2)

        if len(vertices_px) < 3:
            continue

        # Conversão de Pixels para Metros usando a relação exata metros_por_pixel da mesa calibrada
        mpp_x = cfg.largura_mesa_m / float(largura_px) if largura_px > 0 else cfg.metros_por_pixel
        mpp_y = cfg.altura_mesa_m / float(altura_px) if altura_px > 0 else cfg.metros_por_pixel
        vertices_m = np.zeros_like(vertices_px, dtype=np.float64)
        vertices_m[:, 0] = vertices_px[:, 0] * mpp_x  # X em metros

        if cfg.origem_cartesiana_inferior:
            # Inverte o eixo Y para que (0,0) fique no canto inferior esquerdo da mesa
            vertices_m[:, 1] = (altura_px - vertices_px[:, 1]) * mpp_y
        else:
            vertices_m[:, 1] = vertices_px[:, 1] * mpp_y

        lista_poligonos_m.append(np.round(vertices_m, 4))

        # --- Desenho dos obstáculos na imagem de debug ---
        cv2.drawContours(img_debug, [vertices_px], -1, (0, 255, 0), 2)
        for vx, vy in vertices_px:
            cv2.circle(img_debug, (vx, vy), 5, (0, 0, 255), -1)

        # Escreve o ID do polígono e número de vértices no centro do objeto
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            cv2.putText(
                img_debug, f"Obj {len(lista_poligonos_m)} ({len(vertices_px)}v)",
                (cx - 35, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
            )

    # --- Desenho do Robô na imagem de debug ---
    if dados_robo is not None:
        rx, ry = dados_robo["centro_px"]
        r_px = dados_robo["raio_px"]
        cv2.circle(img_debug, (rx, ry), r_px, (0, 0, 255), 2)
        cv2.circle(img_debug, (rx, ry), 5, (0, 0, 255), -1)

        eixo_px = dados_robo.get("eixo_px")
        if eixo_px is not None:
            cv2.circle(img_debug, eixo_px, 6, (255, 0, 255), -1)
            cv2.circle(img_debug, eixo_px, int(0.11 * cfg.pixels_por_metro), (255, 0, 255), 1)

        ang_img = -dados_robo["theta_rad"] if cfg.origem_cartesiana_inferior else dados_robo["theta_rad"]
        castor = dados_robo.get("castor_px")
        if castor is not None:
            cx_c, cy_c = castor
            cv2.circle(img_debug, (cx_c, cy_c), 10, (255, 255, 0), 2)
            cv2.circle(img_debug, (cx_c, cy_c), 4, (0, 255, 255), -1)
            cv2.arrowedLine(img_debug, (rx, ry), (cx_c, cy_c), (0, 255, 0), 3, tipLength=0.25)
        else:
            ponta_x = int(rx + (r_px + 50) * np.cos(ang_img))
            ponta_y = int(ry + (r_px + 50) * np.sin(ang_img))
            cv2.arrowedLine(img_debug, (rx, ry), (ponta_x, ponta_y), (0, 255, 0), 3, tipLength=0.3)

        cv2.putText(
            img_debug, f"ROBO ({dados_robo['x_m']:.2f}m, {dados_robo['y_m']:.2f}m) Th: {dados_robo['theta_graus']:.1f}deg",
            (max(10, rx - 90), max(20, ry - r_px - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
        )

    return lista_poligonos_m, img_debug, mascara


# ==============================================================================
# 4. CAPTURA VIA WEBCAM (Iriun / USB)
# ==============================================================================
def listar_cameras_windows() -> List[str]:
    """Nomes das câmeras DirectShow na ordem dos índices do OpenCV (requer 'pip install pygrabber')."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        return FilterGraph().get_input_devices()
    except Exception:
        return []


def _abrir_camera(indice: int) -> Optional[cv2.VideoCapture]:
    """Abre o índice e confirma que ele realmente entrega frames (o Iriun demora para começar)."""
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(indice, backend)
        if cap.isOpened():
            for _ in range(30):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    return cap
                cv2.waitKey(100)
        cap.release()
    return None


def resolver_camera(cam) -> int:
    """
    Converte o argumento --cam em índice numérico. Aceita:
      - número ("0", "1", ...)  -> usado direto (ANTES era passado como texto ao OpenCV e sempre falhava)
      - nome ("Iriun")          -> procura entre as câmeras do Windows; sem pygrabber, testa os índices 0-9
    """
    if isinstance(cam, int):
        return cam
    cam = str(cam).strip()
    if cam.lstrip("-").isdigit():
        return int(cam)

    nomes = listar_cameras_windows()
    for i, n in enumerate(nomes):
        if cam.lower() in n.lower():
            print(f"[INFO] Câmera '{n}' encontrada no índice {i}.")
            return i

    print(f"[AVISO] Não consegui localizar '{cam}' pelo nome "
          f"({'câmeras: ' + str(nomes) if nomes else 'instale com: pip install pygrabber'}). "
          "Testando índices 0-9...")
    for i in range(10):
        cap = _abrir_camera(i)
        if cap is not None:
            cap.release()
            print(f"[INFO] Usando o índice {i}. Se for a câmera errada, use --cam <número>.")
            return i
    raise RuntimeError("Nenhuma câmera entregou imagem. Abra o app Iriun no celular e o Iriun Webcam no PC.")


def capturar_frame(camera) -> np.ndarray:
    """
    Abre um preview ao vivo e devolve o frame congelado quando o usuário
    pressiona ESPAÇO. ESC cancela. 'camera' pode ser índice ou nome.
    """
    indice_camera = resolver_camera(camera)
    cap = _abrir_camera(indice_camera)
    if cap is None:
        raise RuntimeError(
            f"Não foi possível abrir a câmera de índice {indice_camera} (ou ela não envia imagem). "
            "Verifique se o Iriun está transmitindo no celular e se nenhum outro programa "
            "(Zoom/navegador/outro Python) está usando a câmera. "
            "Rode 'python varredura.py' para ver os índices disponíveis."
        )

    nome_janela = "Captura: ESPACO para congelar | ESC para cancelar"
    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
    print("\n[INSTRUÇÃO] Enquadre a mesa e pressione ESPAÇO para capturar.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if cv2.waitKey(20) & 0xFF == 27:
                    raise RuntimeError("Captura cancelada pelo usuário.")
                continue  # frame perdido (comum no Iriun): tenta o próximo

            cv2.imshow(nome_janela, frame)
            tecla = cv2.waitKey(20) & 0xFF
            if tecla == 32:
                return frame.copy()
            if tecla == 27:
                raise RuntimeError("Captura cancelada pelo usuário.")
    finally:
        cap.release()
        cv2.destroyWindow(nome_janela)


def corrigir_distorcao(imagem: np.ndarray, caminho_npz: str) -> np.ndarray:
    """
    Aplica a correção de distorção da lente usando K e dist salvos em um .npz
    (gerado por calibracao_camera.py). Se o arquivo não existir, devolve a
    imagem original. Se a resolução do frame difere da usada na calibração,
    K é reescalado.
    """
    if not os.path.isabs(caminho_npz) and not os.path.exists(caminho_npz):
        caminho_npz = os.path.join(os.path.dirname(os.path.abspath(__file__)), caminho_npz)
    if not os.path.exists(caminho_npz):
        print(f"[AVISO] Calibração da câmera '{caminho_npz}' não encontrada; imagem sem correção de distorção.")
        return imagem

    dados = np.load(caminho_npz)
    K = dados["K"].astype(np.float64)
    dist = dados["dist"].astype(np.float64)

    alt, larg = imagem.shape[:2]
    if "image_size" in dados.files:
        larg_cal, alt_cal = (int(v) for v in dados["image_size"])
        if (larg_cal, alt_cal) != (larg, alt):
            K[0, :] *= larg / larg_cal
            K[1, :] *= alt / alt_cal
            print(f"[INFO] Resolução ({larg}x{alt}) difere da calibração da câmera ({larg_cal}x{alt_cal}); K reescalado.")

    corrigida = cv2.undistort(imagem, K, dist)
    print("[INFO] Distorção da lente corrigida com", os.path.basename(caminho_npz))
    return corrigida


# ==============================================================================
# 5. EXECUÇÃO PRINCIPAL
# ==============================================================================
def processar_mapa(imagem: np.ndarray, cfg: ConfigMapa):
    # 1. Obter os 4 cantos da mesa
    if cfg.cantos_calibrados is not None:
        # Cantos vindos da calibração com tabuleiro (já ordenados, NÃO reordenar)
        alt_img, larg_img = imagem.shape[:2]
        cantos = np.array(cfg.cantos_calibrados, dtype="float32")
        if (larg_img, alt_img) != (cfg.largura_camera_px, cfg.altura_camera_px) and cfg.largura_camera_px > 0 and cfg.altura_camera_px > 0:
            sx = larg_img / cfg.largura_camera_px
            sy = alt_img / cfg.altura_camera_px
            cantos[:, 0] *= sx
            cantos[:, 1] *= sy
            print(f"[INFO] Resolução da foto ({larg_img}x{alt_img}) adaptada à calibração "
                  f"({cfg.largura_camera_px}x{cfg.altura_camera_px}) por escala ({sx:.3f}x, {sy:.3f}y).")
    elif cfg.pontos_mesa_fixos is not None and len(cfg.pontos_mesa_fixos) == 4:
        cantos = ordenar_4_pontos(np.array(cfg.pontos_mesa_fixos, dtype="float32"))
    elif cfg.auto_detectar_mesa:
        cantos = detectar_mesa_auto(imagem)
    else:
        cantos = selecionar_mesa_manual(imagem)

    # Atualiza a escala (pixels_por_metro e metros_por_pixel) diretamente a partir
    # dos cantos da mesa na imagem atual e das dimensões reais calibradas (2.50 m x 1.21 m)
    escala_img = calcular_escala_cantos_px(cantos, cfg.largura_mesa_m, cfg.altura_mesa_m)
    cfg.pixels_por_metro = escala_img["pixels_por_metro"]
    cfg.metros_por_pixel = escala_img["metros_por_pixel"]
    print(f"[INFO] Escala da imagem atual: {cfg.pixels_por_metro:.2f} px/m -> "
          f"1 pixel = {cfg.metros_por_pixel * 1000:.4f} mm ({cfg.metros_por_pixel:.6f} m/px)")

    # 2. Isolar e retificar a mesa
    mesa_retificada = retificar_mesa(imagem, cantos, cfg)
    altura_ret_px, largura_ret_px = mesa_retificada.shape[:2]

    # 2b. Máscara do interior da fita preta (formato de estádio), se calibrado
    mascara_mesa = None
    contorno_ret_px = None
    if cfg.contorno_mesa_px is not None:
        contorno_mesa = np.array(cfg.contorno_mesa_px, dtype="float32")
        alt_img, larg_img = imagem.shape[:2]
        if (larg_img, alt_img) != (cfg.largura_camera_px, cfg.altura_camera_px) and cfg.largura_camera_px > 0 and cfg.altura_camera_px > 0:
            contorno_mesa[:, 0] *= (larg_img / cfg.largura_camera_px)
            contorno_mesa[:, 1] *= (alt_img / cfg.altura_camera_px)
        contorno_ret_px = aplicar_homografia(contorno_mesa, matriz_retificacao(cantos, cfg))
        mascara_mesa = np.zeros(mesa_retificada.shape[:2], np.uint8)
        cv2.fillPoly(mascara_mesa, [np.round(contorno_ret_px).astype(np.int32)], 255)

    # 3. Detectar o robô (vermelho/laranja) em HSV e gerar máscara de exclusão
    dados_robo, mascara_exclusao_robo = detectar_robo_hsv(mesa_retificada, cfg, mascara_mesa)

    # 4. Extrair matrizes de vértices (x, y) dos obstáculos (sem incluir o robô)
    poligonos_m, img_debug, mascara = extrair_obstaculos_poligonais(
        mesa_retificada, cfg, mascara_exclusao_robo, dados_robo, mascara_mesa
    )
    if contorno_ret_px is not None:
        cv2.polylines(img_debug, [np.round(contorno_ret_px).astype(np.int32)], True, (255, 0, 255), 2)

    # 5. Exibir resultados no terminal e exportar JSON
    print(f"\n================ MAPA EXTRAÍDO ({cfg.largura_mesa_m}m x {cfg.altura_mesa_m}m) ================")
    if dados_robo is not None:
        print(
            f"[ROBÔ DETECTADO] Posição: (x = {dados_robo['x_m']} m, y = {dados_robo['y_m']} m) | "
            f"Theta = {dados_robo['theta_graus']}° | Raio = {dados_robo['raio_marcador_m']} m"
        )
        robo_json = {
            "x_m": dados_robo["x_m"],
            "y_m": dados_robo["y_m"],
            "marcador_x_m": dados_robo["marcador_x_m"],
            "marcador_y_m": dados_robo["marcador_y_m"],
            "offset_centro_m": dados_robo["offset_centro_m"],
            "theta_rad": dados_robo["theta_rad"],
            "theta_graus": dados_robo["theta_graus"],
            "raio_marcador_m": dados_robo["raio_marcador_m"],
            "vertices_xy_m": dados_robo["vertices_xy_m"].tolist()
        }
    else:
        print("[AVISO] Robô (marcador vermelho/laranja) NÃO foi detectado na mesa!")
        robo_json = None

    print(f"Total de obstáculos detectados: {len(poligonos_m)}\n")

    dados_exportacao = {
        "workspace_limites_m": {
            "x_min": 0.0,
            "x_max": cfg.largura_mesa_m,
            "y_min": 0.0,
            "y_max": cfg.altura_mesa_m,
            "origem": "inferior_esquerdo" if cfg.origem_cartesiana_inferior else "superior_esquerdo"
        },
        "escala_calibrada": {
            "pixels_por_metro": round(cfg.pixels_por_metro, 4),
            "metros_por_pixel": round(cfg.metros_por_pixel, 6),
            "resolucao_retificada_px": [int(largura_ret_px), int(altura_ret_px)],
            "metros_por_pixel_x": round(cfg.largura_mesa_m / float(largura_ret_px), 6),
            "metros_por_pixel_y": round(cfg.altura_mesa_m / float(altura_ret_px), 6),
        },
        "robo": robo_json,
        "obstaculos": []
    }

    # Contorno real da mesa (interior da fita, formato de estádio) em metros
    if contorno_ret_px is not None:
        dados_exportacao["workspace_contorno_xy_m"] = [
            list(pixel_para_metro(x, y, altura_ret_px, cfg, largura_ret_px)) for x, y in contorno_ret_px
        ]

    for i, poly in enumerate(poligonos_m):
        print(f"Obstáculo {i+1} (Matriz {poly.shape[0]}x2 em metros):")
        print(poly)
        print("-" * 45)
        dados_exportacao["obstaculos"].append({
            "id": i + 1,
            "num_vertices": int(poly.shape[0]),
            "vertices_xy_m": poly.tolist()
        })

    with open("mapa_obstaculos.json", "w", encoding="utf-8") as f:
        json.dump(dados_exportacao, f, indent=4)
    print("[OK] Robô e matrizes de obstáculos salvos em 'mapa_obstaculos.json'")

    # 6. Visualização final (sempre salva as imagens de conferência em disco)
    cv2.imwrite("mapa_debug_mascara.png", mascara)
    cv2.imwrite("mapa_debug_retificada.png", img_debug)
    print("[OK] Imagens de conferência salvas em 'mapa_debug_mascara.png' e 'mapa_debug_retificada.png'")
    if not cfg.sem_gui:
        cv2.imshow("1. Mascara Binaria (Obstaculos sem o Robo)", mascara)
        cv2.imshow("2. Mesa Retificada, Robo e Vertices Detectados", img_debug)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return dados_robo, poligonos_m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img", help="Caminho/nome da imagem", type=str, nargs="?")
    parser.add_argument("-larg", help="largura da mesa em metros (x)", type=float, default=LARGURA_MESA_PADRAO_M)
    parser.add_argument('-altu', help="altura da mesa em metros (y)", type=float, default=ALTURA_MESA_PADRAO_M)
    parser.add_argument('--ac', help="auto calibração (s/y/n)", type=str, default='y')
    parser.add_argument('--ib', help="inverter cores (padrão mesa escura) (s/y/n)", type=str, default='y')
    parser.add_argument('--cam', help="captura da webcam buscando o device pelo nome", type=str,
                        nargs="?", const="Iriun", metavar="NOME")

    # --- Calibração com tabuleiro quadriculado ---
    parser.add_argument('--calibrar', action='store_true',
                        help="modo calibração: foto com o tabuleiro SOZINHO sobre a mesa")
    parser.add_argument('--tab', type=int, nargs=2, metavar=("QX", "QY"),
                        help=f"número de QUADRADOS do tabuleiro em cada direção (padrão: {TABULEIRO_QUADRADOS[0]} {TABULEIRO_QUADRADOS[1]})")
    parser.add_argument('--lado', type=float, help=f"lado de cada quadrado do tabuleiro em metros (padrão: {TABULEIRO_LADO_M})")
    parser.add_argument('--calib-arq', default="calibracao_mesa.json", help="arquivo de calibração")
    parser.add_argument('--sem-calib', action='store_true', help="ignora o arquivo de calibração existente")
    parser.add_argument('--cam-calib', default="calibracao_camera.npz",
                        help="arquivo .npz (K, dist, image_size) gerado por calibracao_camera.py")
    parser.add_argument('--sem-undistort', action='store_true',
                        help="não corrige a distorção da lente")
    parser.add_argument('--sem-gui', action='store_true',
                        help="não abre janelas (salva as imagens de conferência em arquivos .png)")
    parser.add_argument('--limiar-fita', type=int, default=60,
                        help="brilho (V do HSV, 0-255) máximo da fita preta (padrão 60)")
    parser.add_argument('--offset-theta', type=float, default=0.0,
                        help="correção angular adicional somada ao theta do robô em graus")
    parser.add_argument('--offset-centro', type=float, default=0.075,
                        help="distância em metros do marcador vermelho traseiro ao centro do eixo das rodas (padrão 0.075)")
    parser.add_argument('--limiar-corpo', type=int, default=90,
                        help="brilho (V do HSV, 0-255) máximo para segmentação das peças/castor do robô (padrão 90)")

    args = parser.parse_args()

    if not args.img and not args.cam:
        parser.error("informe o caminho de uma imagem ou use --cam")
    if args.tab is None:
        args.tab = list(TABULEIRO_QUADRADOS)
    if args.lado is None:
        args.lado = TABULEIRO_LADO_M

    ad = True if (args.ac.lower() == 's' or args.ac.lower() == 'y') else False
    ib = True if (args.ib.lower() == 's' or args.ib.lower() == 'y') else False

    # Configure aqui os parâmetros reais da sua bancada
    config = ConfigMapa(
        largura_mesa_m=args.larg,          # Altere para a largura real (ou 1.0 se quiser normalizar como quadrado)
        altura_mesa_m=args.altu,           # Altere para a profundidade real (ou 1.0 se quiser normalizar como quadrado)
        auto_detectar_mesa=ad,             # False = abre a janela para clicar nos 4 cantos da mesa
        inverter_binarizacao=ib,           # Mude para True se a mesa for clara e os objetos escuros
        limiar_fita_v=args.limiar_fita,
        limiar_corpo_robo_v=args.limiar_corpo,
        offset_theta_robo_graus=args.offset_theta,
        offset_centro_robo_m=args.offset_centro,
        sem_gui=args.sem_gui
    )
    
    if args.cam:
        imagem_entrada = capturar_frame(args.cam)
    else:
        imagem_entrada = cv2.imread(args.img)
        if imagem_entrada is None:
            raise FileNotFoundError(f"Não foi possível carregar a imagem: {args.img}")

    # Remove a distorção da lente com os parâmetros intrínsecos da câmera
    if not args.sem_undistort:
        imagem_entrada = corrigir_distorcao(imagem_entrada, args.cam_calib)

    if args.calibrar:
        calibracao = calibrar_com_tabuleiro(imagem_entrada, args.tab[0], args.tab[1], args.lado, config)
        if calibracao is not None:
            salvar_calibracao(calibracao, args.calib_arq)
    else:
        if not args.sem_calib and carregar_calibracao(args.calib_arq, config):
            print("[INFO] -larg/-altu e --ac ignorados (usando dimensões e cantos da calibração).")
        processar_mapa(imagem_entrada, config)
