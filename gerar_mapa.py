import cv2
import numpy as np
import json
import glob
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import argparse

# ==============================================================================
# 1. CONFIGURAÇÕES DO LABORATÓRIO E PARÂMETROS DA MESA
# ==============================================================================
@dataclass
class ConfigMapa:
    # Dimensões reais da mesa no sistema métrico (altere para o tamanho real medido)
    # Se quiser tratar como um quadrado perfeito de 1x1m, basta usar 1.0 e 1.0
    largura_mesa_m: float = 1.20   # Eixo X em metros
    altura_mesa_m: float = 0.80    # Eixo Y em metros
    largura_camera_px: int = 1280
    altura_camera_px: int = 720

    # Resolução interna da imagem retificada (ex: 600 px/m -> 1 pixel = 1.66 mm)
    pixels_por_metro: int = 600
    
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
    origem_cartesiana_inferior: bool = True

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
    margem_exclusao_robo: float = 0.09 # Raio extra (em metros) para apagar o robô da máscara de obstáculos


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
# 3. RETIFICAÇÃO, DETECÇÃO DO ROBÔ (HSV) E EXTRAÇÃO DOS POLÍGONOS
# ==============================================================================
def retificar_mesa(imagem: np.ndarray, cantos_mesa: np.ndarray, cfg: ConfigMapa) -> np.ndarray:
    """
    Aplica Homografia para recortar apenas a mesa e deixá-la na proporção métrica exata.
    """
    largura_px = int(cfg.largura_mesa_m * cfg.pixels_por_metro)
    altura_px = int(cfg.altura_mesa_m * cfg.pixels_por_metro)

    destino = np.array([
        [0, 0],
        [largura_px - 1, 0],
        [largura_px - 1, altura_px - 1],
        [0, altura_px - 1]
    ], dtype="float32")

    matriz_H = cv2.getPerspectiveTransform(cantos_mesa, destino)
    mesa_retificada = cv2.warpPerspective(imagem, matriz_H, (largura_px, altura_px))
    return mesa_retificada


def pixel_para_metro(x_px: float, y_px: float, altura_px: int, cfg: ConfigMapa) -> Tuple[float, float]:
    """Converte coordenadas de pixel da imagem retificada para metros (x, y)."""
    x_m = x_px / cfg.pixels_por_metro
    if cfg.origem_cartesiana_inferior:
        y_m = (altura_px - y_px) / cfg.pixels_por_metro
    else:
        y_m = y_px / cfg.pixels_por_metro
    return round(float(x_m), 4), round(float(y_m), 4)


def detectar_robo_hsv(mesa_img: np.ndarray, cfg: ConfigMapa) -> Tuple[Optional[Dict], np.ndarray]:
    """
    Localiza o marcador vermelho/laranja do robô na imagem retificada da mesa.
    Retorna:
      - dados_robo: Dicionário com (x_m, y_m, theta_rad, theta_graus, vertices_xy_m, etc.) ou None
      - mascara_exclusao_robo: Máscara binária dilatada da região do robô para removê-lo dos obstáculos
    """
    altura_px, _ = mesa_img.shape[:2]
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

    # Centroide do robô em pixels e em metros
    cx_px = M["m10"] / M["m00"]
    cy_px = M["m01"] / M["m00"]
    robo_x_m, robo_y_m = pixel_para_metro(cx_px, cy_px, altura_px, cfg)

    # Raio do marcador e orientação pelo eixo principal de inércia (momentos centrais)
    (_, _), raio_px = cv2.minEnclosingCircle(maior_contorno)
    raio_m = round(float(raio_px / cfg.pixels_por_metro), 4)

    mu20 = M["mu20"] / M["m00"]
    mu02 = M["mu02"] / M["m00"]
    mu11 = M["mu11"] / M["m00"]
    theta_img_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    theta_rad = -theta_img_rad if cfg.origem_cartesiana_inferior else theta_img_rad

    # Vértices aproximados do marcador do robô
    perimetro = cv2.arcLength(maior_contorno, True)
    vertices_robo_px = cv2.approxPolyDP(maior_contorno, cfg.epsilon_poligono * perimetro, True).reshape(-1, 2)
    vertices_robo_m = np.array([pixel_para_metro(vx, vy, altura_px, cfg) for vx, vy in vertices_robo_px])

    # Cria círculo dilatado na máscara de exclusão para apagar o robô da máscara de obstáculos
    margem_exclusao_px = int(cfg.margem_exclusao_robo * cfg.pixels_por_metro)
    cv2.circle(mascara_exclusao, (int(cx_px), int(cy_px)), int(raio_px + margem_exclusao_px), 255, -1)

    dados_robo = {
        "x_m": robo_x_m,
        "y_m": robo_y_m,
        "theta_rad": round(float(theta_rad), 4),
        "theta_graus": round(float(np.degrees(theta_rad)), 2),
        "raio_marcador_m": raio_m,
        "vertices_xy_m": vertices_robo_m,
        "centro_px": (int(cx_px), int(cy_px)),
        "raio_px": int(raio_px)
    }

    return dados_robo, mascara_exclusao


def extrair_obstaculos_poligonais(
    mesa_img: np.ndarray,
    cfg: ConfigMapa,
    mascara_exclusao_robo: Optional[np.ndarray] = None,
    dados_robo: Optional[Dict] = None
):
    """
    Identifica os objetos sobre a mesa, extrai seus vértices e converte para metros.
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
    _, mascara = cv2.threshold(blur, 0, 255, tipo_thresh + cv2.THRESH_OTSU)

    # Remove a região do robô para que ele NÃO seja detectado como obstáculo
    if mascara_exclusao_robo is not None:
        mascara = cv2.bitwise_and(mascara, cv2.bitwise_not(mascara_exclusao_robo))

    # Zera as bordas da imagem para evitar que rebarbas da quina da mesa virem obstáculos
    margem_px = int(cfg.margem_borda_m * cfg.pixels_por_metro)
    if margem_px > 0:
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

        # Conversão de Pixels para Metros
        vertices_m = np.zeros_like(vertices_px, dtype=np.float64)
        vertices_m[:, 0] = vertices_px[:, 0] / cfg.pixels_por_metro # X em metros

        if cfg.origem_cartesiana_inferior:
            # Inverte o eixo Y para que (0,0) fique no canto inferior esquerdo da mesa
            vertices_m[:, 1] = (altura_px - vertices_px[:, 1]) / cfg.pixels_por_metro
        else:
            vertices_m[:, 1] = vertices_px[:, 1] / cfg.pixels_por_metro

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
        cv2.circle(img_debug, (rx, ry), r_px, (255, 255, 0), 2)
        cv2.circle(img_debug, (rx, ry), 5, (0, 140, 255), -1)

        ang_img = -dados_robo["theta_rad"] if cfg.origem_cartesiana_inferior else dados_robo["theta_rad"]
        ponta_x = int(rx + (r_px + 15) * np.cos(ang_img))
        ponta_y = int(ry + (r_px + 15) * np.sin(ang_img))
        cv2.arrowedLine(img_debug, (rx, ry), (ponta_x, ponta_y), (255, 255, 0), 2, tipLength=0.3)

        cv2.putText(
            img_debug, f"ROBO ({dados_robo['x_m']:.2f}m, {dados_robo['y_m']:.2f}m)",
            (max(10, rx - 70), max(20, ry - r_px - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
        )

    return lista_poligonos_m, img_debug, mascara


# ==============================================================================
# 4. CAPTURA VIA WEBCAM (Iriun / USB)
# ==============================================================================
def localizar_camera_por_nome(nome_busca: str) -> int:
    """
    Procura em /sys/class/video4linux um device cujo nome contenha 'nome_busca'
    e retorna seu índice (ex: 2 para /dev/video2).
    """
    candidatos = []
    for caminho_nome in sorted(glob.glob("/sys/class/video4linux/video*/name")):
        try:
            with open(caminho_nome, encoding="utf-8") as f:
                nome_device = f.read().strip()
        except OSError:
            continue
        if nome_busca.lower() in nome_device.lower():
            indice = int(re.search(r"/video(\d+)/name$", caminho_nome).group(1))
            candidatos.append((indice, nome_device))

    if not candidatos:
        raise RuntimeError(
            f"Nenhuma câmera com '{nome_busca}' no nome foi encontrada. "
            "Abra o app 'iriunwebcam' no PC e conecte o celular antes de rodar."
        )

    indice, nome_device = candidatos[0]
    print(f"[INFO] Câmera encontrada: /dev/video{indice} ({nome_device})")
    return indice


def capturar_frame(indice_camera: int) -> np.ndarray:
    """
    Abre um preview ao vivo e devolve o frame congelado quando o usuário
    pressiona ESPAÇO. ESC cancela.
    """
    cap = cv2.VideoCapture(indice_camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Não foi possível abrir /dev/video{indice_camera}. "
            "Verifique se o celular está transmitindo no app do Iriun."
        )

    nome_janela = "Captura: ESPACO para congelar | ESC para cancelar"
    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
    print("\n[INSTRUÇÃO] Enquadre a mesa e pressione ESPAÇO para capturar.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Falha ao ler frame da câmera.")

            cv2.imshow(nome_janela, frame)
            tecla = cv2.waitKey(20) & 0xFF
            if tecla == 32:
                return frame.copy()
            if tecla == 27:
                raise RuntimeError("Captura cancelada pelo usuário.")
    finally:
        cap.release()
        cv2.destroyWindow(nome_janela)


# ==============================================================================
# 5. EXECUÇÃO PRINCIPAL
# ==============================================================================
def processar_mapa(imagem: np.ndarray, cfg: ConfigMapa):
    # 1. Obter os 4 cantos da mesa
    if cfg.pontos_mesa_fixos is not None and len(cfg.pontos_mesa_fixos) == 4:
        cantos = ordenar_4_pontos(np.array(cfg.pontos_mesa_fixos, dtype="float32"))
    elif cfg.auto_detectar_mesa:
        cantos = detectar_mesa_auto(imagem)
    else:
        cantos = selecionar_mesa_manual(imagem)

    # 2. Isolar e retificar a mesa
    mesa_retificada = retificar_mesa(imagem, cantos, cfg)

    # 3. Detectar o robô (vermelho/laranja) em HSV e gerar máscara de exclusão
    dados_robo, mascara_exclusao_robo = detectar_robo_hsv(mesa_retificada, cfg)

    # 4. Extrair matrizes de vértices (x, y) dos obstáculos (sem incluir o robô)
    poligonos_m, img_debug, mascara = extrair_obstaculos_poligonais(
        mesa_retificada, cfg, mascara_exclusao_robo, dados_robo
    )

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
        "robo": robo_json,
        "obstaculos": []
    }

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

    # 6. Visualização final
    cv2.imshow("1. Mascara Binaria (Obstaculos sem o Robo)", mascara)
    cv2.imshow("2. Mesa Retificada, Robo e Vertices Detectados", img_debug)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return dados_robo, poligonos_m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img", help="Caminho/nome da imagem", type=str, nargs="?")
    parser.add_argument("-larg", help="largura da mesa em metros (x)", type=float, default=1.2)
    parser.add_argument('-altu', help="altura da mesa em metros (y)", type=float, default=0.8)
    parser.add_argument('--ac', help="auto calibração (s/y/n)", type=str, default='y')
    parser.add_argument('--ib', help="inverter cores (padrão mesa escura) (s/y/n)", type=str, default='y')
    parser.add_argument('--cam', help="captura da webcam buscando o device pelo nome", type=str,
                        nargs="?", const="Iriun", metavar="NOME")

    args = parser.parse_args()

    if not args.img and not args.cam:
        parser.error("informe o caminho de uma imagem ou use --cam")

    ad = True if (args.ac.lower() == 's' or args.ac.lower() == 'y') else False
    ib = True if (args.ib.lower() == 's' or args.ib.lower() == 'y') else False

    # Configure aqui os parâmetros reais da sua bancada
    config = ConfigMapa(
        largura_mesa_m=args.larg,          # Altere para a largura real (ou 1.0 se quiser normalizar como quadrado)
        altura_mesa_m=args.altu,           # Altere para a profundidade real (ou 1.0 se quiser normalizar como quadrado)
        auto_detectar_mesa=ad,             # False = abre a janela para clicar nos 4 cantos da mesa
        inverter_binarizacao=ib            # Mude para True se a mesa for clara e os objetos escuros
    )
    
    if args.cam:
        imagem_entrada = capturar_frame(localizar_camera_por_nome(args.cam))
    else:
        imagem_entrada = cv2.imread(args.img)
        if imagem_entrada is None:
            raise FileNotFoundError(f"Não foi possível carregar a imagem: {args.img}")

    processar_mapa(imagem_entrada, config)