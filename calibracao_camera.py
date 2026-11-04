import os
import glob
import cv2
import numpy as np

# Parâmetros do tabuleiro de xadrez
# Tabuleiro de 10x7 quadrados => 9x6 cantos internos (largura x altura)
chessboard_size = (9, 6)
square_size = 0.025  # tamanho do quadrado (altere p/ unidade real, ex.: 0.025 m)

# Fator de ampliação: o tabuleiro nas imagens é muito pequeno (~45 px),
# então ampliamos a imagem para a detecção funcionar.
SCALE = 3

# Preparar pontos do objeto 3D
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

object_points = []
image_points = []

pasta = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'chessboard')
list_of_image_files = sorted(glob.glob(os.path.join(pasta, '*.png')))

image_size = None
MOSTRAR = os.environ.get('DISPLAY') is not None

for image_file in list_of_image_files:
    image = cv2.imread(image_file)
    if image is None:
        print(f"Não foi possível ler {image_file}")
        continue
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    big = cv2.resize(gray, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    ret, corners = cv2.findChessboardCornersSB(
        big, chessboard_size, cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
    )

    if not ret:
        print(f"[falhou] {os.path.basename(image_file)}")
        continue

    # Voltar para a escala original (centro do pixel: (x + 0.5)/SCALE - 0.5)
    corners = ((corners + 0.5) / SCALE - 0.5).astype(np.float32)
    print(f"[ok]     {os.path.basename(image_file)}")

    object_points.append(objp)
    image_points.append(corners)

    if MOSTRAR:
        vis = image.copy()
        cv2.drawChessboardCorners(vis, chessboard_size, corners, ret)
        vis = cv2.resize(vis, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('img', vis)
        cv2.waitKey(500)

cv2.destroyAllWindows()

if len(object_points) < 3:
    raise SystemExit(f"Poucas imagens com tabuleiro detectado ({len(object_points)}). "
                     "São necessárias pelo menos 3 (ideal 10+).")

# Calibrar a câmera
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, image_size, None, None,
    # Poucas imagens/tabuleiro pequeno: modelo simplificado evita valores absurdos
    flags=cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_PRINCIPAL_POINT
)

print(f"\nImagens usadas: {len(object_points)}/{len(list_of_image_files)}")
print("Erro RMS de reprojeção:", ret)
print("Matriz de calibração K:\n", K)
print("Distorção:", dist.ravel())

np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibracao_camera.npz'),
         K=K, dist=dist, image_size=image_size)
