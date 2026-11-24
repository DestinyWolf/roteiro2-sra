import numpy as np


def norm_angle(angulo):
    """Normaliza o ângulo (rad) para [-pi, pi). Funciona com escalar ou array."""
    return (angulo + np.pi) % (2 * np.pi) - np.pi


# ==============================================================================
# Predição (modelo de odometria: o robô anda delta_s e gira delta_theta)
# Estado: mu = [x, y, theta]
# ==============================================================================
def ekf_pred(mu, sigma, delta_s, delta_theta, R):
    x, y, theta = mu

    theta_meio = theta + delta_theta / 2
    x_bar = x + delta_s * np.cos(theta_meio)
    y_bar = y + delta_s * np.sin(theta_meio)
    theta_bar = norm_angle(theta + delta_theta)

    mu_bar = np.array([x_bar, y_bar, theta_bar])

    # Jacobiano do modelo de movimento em relação ao estado
    G = np.array([
        [1, 0, -delta_s * np.sin(theta_meio)],
        [0, 1,  delta_s * np.cos(theta_meio)],
        [0, 0, 1]
    ])

    sigma_bar = G @ sigma @ G.T + R

    return mu_bar, sigma_bar


# ==============================================================================
# Correção (medição da câmera: observa diretamente [x, y, theta])
# ==============================================================================
def ekf_corr(mu_bar, sigma_bar, x_cam, y_cam, theta_cam_graus, Q):
    I = np.identity(3)

    theta_cam = norm_angle(np.deg2rad(theta_cam_graus))
    z = np.array([x_cam, y_cam, theta_cam])

    H = I  # a câmera mede o estado diretamente

    z_esp = H @ mu_bar
    inovacao = z - z_esp
    inovacao[2] = norm_angle(inovacao[2])  # menor diferença angular

    S = H @ sigma_bar @ H.T + Q
    K = sigma_bar @ H.T @ np.linalg.inv(S)

    mu = mu_bar + K @ inovacao
    mu[2] = norm_angle(mu[2])

    # Forma de Joseph: mantém sigma simétrica e positiva definida
    A = I - K @ H
    sigma = A @ sigma_bar @ A.T + K @ Q @ K.T

    return mu, sigma


def ekf_step(mu, sigma, delta_s, delta_theta, x_cam, y_cam, theta_cam_graus, mapa, R, Q):
    """
    Um passo completo do EKF. Se a câmera não enxergou o robô (x_cam is None),
    apenas a predição é usada. 'mapa' fica reservado (ex.: rejeitar medições dentro de obstáculos).
    """
    mu_bar, sigma_bar = ekf_pred(mu, sigma, delta_s, delta_theta, R)

    if x_cam is not None and y_cam is not None and theta_cam_graus is not None:
        mu, sigma = ekf_corr(mu_bar, sigma_bar, x_cam, y_cam, theta_cam_graus, Q)
    else:
        mu, sigma = mu_bar, sigma_bar

    return mu, sigma


    return mu, sigma