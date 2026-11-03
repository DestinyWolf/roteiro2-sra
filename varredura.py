import cv2

for index in range(5):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"Câmera encontrada e funcionando no índice: {index}")
        cap.release()
    else:
        print(f"Índice {index} está fechado ou ocupado.")