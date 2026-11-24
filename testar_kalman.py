"""
Teste do Filtro de Kalman Estendido (kalman_filter.py) com dados fictícios.

1) Testes unitários (normalização de ângulo, predição, correção, dropout da câmera)
2) Simulação: o robô segue waypoints desviando dos obstáculos de 'mapa_obstaculos.json'
   - odometria ruidosa (+ viés de giro)  -> usada na predição
   - câmera ruidosa com perdas e outliers -> usada na correção
   Compara: odometria pura x câmera pura x EKF e salva um gráfico.

Uso:
    python testar_kalman.py                  # roda tudo e salva teste_kalman.png
    python testar_kalman.py --show           # também abre a janela do gráfico
    python testar_kalman.py --sem-camera     # câmera nunca enxerga (só predição)
    python testar_kalman.py --ruido-cam 0.03 --perda 0.5 --seed 3
"""
import argparse
import json
import os
import numpy as np

import kalman_filter as kf

AQUI = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------------------
# Testes unitários
# ------------------------------------------------------------------------------
def testes_unitarios():
    ok_geral = True

    def check(nome, cond, extra=""):
        nonlocal ok_geral
        ok_geral &= bool(cond)
        print(f"  [{'OK' if cond else 'FALHOU'}] {nome} {extra}")

    print("== Testes unitários ==")

    # norm_angle
    casos = [(0, 0), (np.pi / 2, np.pi / 2), (3 * np.pi / 2, -np.pi / 2),
             (-3 * np.pi / 2, np.pi / 2), (2 * np.pi, 0), (-np.pi * 0.99, -np.pi * 0.99),
             (4 * np.pi + 0.3, 0.3)]
    check("norm_angle em [-pi, pi)",
          all(abs(kf.norm_angle(a) - e) < 1e-9 for a, e in casos))

    # predição: reta
    R = np.diag([1e-4, 1e-4, 1e-4])
    P0 = np.diag([0.01, 0.01, 0.01])
    mu, P = kf.ekf_pred(np.array([0, 0, 0.0]), P0, 0.5, 0.0, R)
    check("predição: andar 0,5 m em theta=0", np.allclose(mu, [0.5, 0, 0]), f"-> {mu}")
    check("predição: incerteza cresce", np.trace(P) > np.trace(P0))
    check("predição: sigma simétrica", np.allclose(P, P.T))

    # predição: theta = 90°
    mu, _ = kf.ekf_pred(np.array([1.0, 1.0, np.pi / 2]), P0, 0.2, 0.0, R)
    check("predição: theta=90° anda em +y", np.allclose(mu, [1.0, 1.2, np.pi / 2]), f"-> {mu}")

    # predição: cruzando ±pi
    mu, _ = kf.ekf_pred(np.array([0, 0, np.pi - 0.05]), P0, 0.0, 0.1, R)
    check("predição: theta cruza +pi sem estourar", -np.pi <= mu[2] < np.pi and abs(mu[2] - (-np.pi + 0.05)) < 1e-9,
          f"-> {mu[2]:.4f}")

    # Jacobiano numérico x analítico
    def f(m):
        return kf.ekf_pred(m, P0, 0.3, 0.2, R)[0]

    m0 = np.array([0.4, 0.7, 0.5])
    J = np.zeros((3, 3))
    for i in range(3):
        d = np.zeros(3); d[i] = 1e-6
        J[:, i] = (f(m0 + d) - f(m0 - d)) / 2e-6
    _, Pb = kf.ekf_pred(m0, P0, 0.3, 0.2, R)
    check("predição: sigma_bar = J P J' + R (Jacobiano correto)", np.allclose(Pb, J @ P0 @ J.T + R, atol=1e-6))

    # correção: medição perfeita com Q pequena -> estado vai para a medição
    mu_bar = np.array([1.0, 1.0, 0.0])
    Pb = np.diag([0.5, 0.5, 0.5])
    mu, P = kf.ekf_corr(mu_bar, Pb, 1.2, 0.9, 10.0, np.diag([1e-6, 1e-6, 1e-6]))
    check("correção: Q pequena -> segue a câmera", np.allclose(mu, [1.2, 0.9, np.deg2rad(10)], atol=1e-3), f"-> {mu}")

    # correção: Q enorme -> ignora a câmera
    mu, P = kf.ekf_corr(mu_bar, Pb, 5.0, 5.0, 90.0, np.diag([1e6, 1e6, 1e6]))
    check("correção: Q enorme -> ignora a câmera", np.allclose(mu, mu_bar, atol=1e-3))

    # correção: incerteza diminui
    mu, P = kf.ekf_corr(mu_bar, Pb, 1.1, 1.0, 0.0, np.diag([0.01, 0.01, 0.01]))
    check("correção: incerteza diminui", np.trace(P) < np.trace(Pb), f"(tr {np.trace(Pb):.3f} -> {np.trace(P):.3f})")
    check("correção: sigma simétrica e positiva definida",
          np.allclose(P, P.T) and np.all(np.linalg.eigvalsh(P) > 0))

    # correção: atravessando ±pi (estado 179°, câmera -179° -> diferença 2°, não 358°)
    mu, _ = kf.ekf_corr(np.array([0, 0, np.deg2rad(179.0)]), Pb, 0, 0, -179.0, np.diag([0.01] * 3))
    check("correção: ângulo 179° vs -179° (menor caminho)", abs(np.rad2deg(mu[2])) > 170, f"-> {np.rad2deg(mu[2]):.1f}°")

    # ekf_step: retorno e dropout
    out = kf.ekf_step(np.array([0, 0, 0.0]), P0, 0.1, 0.0, None, None, None, None, R, np.eye(3) * 0.01)
    check("ekf_step devolve (mu, sigma) com câmera = None", out is not None and len(out) == 2)

    # Convergência de um estado inicial errado
    rng = np.random.default_rng(0)
    verdade = np.array([1.0, 0.5, 0.3])
    mu, P = np.array([1.5, 0.0, -0.5]), np.diag([1.0, 1.0, 1.0])
    Q = np.diag([0.02 ** 2, 0.02 ** 2, np.deg2rad(3) ** 2])
    for _ in range(40):
        zc = verdade + rng.normal(0, np.sqrt(np.diag(Q)))
        mu, P = kf.ekf_step(mu, P, 0.0, 0.0, zc[0], zc[1], np.rad2deg(zc[2]), None, R * 0.01, Q)
    check("convergência: estado inicial errado -> verdade", np.linalg.norm(mu[:2] - verdade[:2]) < 0.03
          and abs(kf.norm_angle(mu[2] - verdade[2])) < 0.08, f"-> {mu}")

    print("RESULTADO:", "TODOS OS TESTES PASSARAM" if ok_geral else "HÁ FALHAS")
    return ok_geral


# ------------------------------------------------------------------------------
# Simulação com o mapa real
# ------------------------------------------------------------------------------
def carregar_mapa(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        d = json.load(f)
    obst = [np.array(o["vertices_xy_m"], float) for o in d["obstaculos"]]
    contorno = np.array(d["workspace_contorno_xy_m"], float) if d.get("workspace_contorno_xy_m") else None
    lim = d["workspace_limites_m"]
    robo = d.get("robo")
    return {"obstaculos": obst, "contorno": contorno, "limites": lim, "robo": robo}


def ponto_em_poligono(p, poly):
    from matplotlib.path import Path
    return Path(poly).contains_point(p)


def colide(p, mapa, raio=0.0):
    from matplotlib.path import Path
    return any(Path(o).contains_point(p, radius=-raio) or Path(o).contains_point(p, radius=raio)
               for o in mapa["obstaculos"])


def waypoints_padrao(mapa):
    """Rota fictícia que contorna os obstáculos do mapa (por cima), a partir da pose do robô."""
    r = mapa["robo"]
    x0, y0 = (r["x_m"], r["y_m"]) if r else (1.8, 0.6)
    return [(x0, y0), (1.80, 1.00), (1.00, 1.00), (0.55, 0.70), (0.55, 0.30), (1.00, 0.25), (1.00, 0.40)]


def simular(mapa, args):
    rng = np.random.default_rng(args.seed)
    dt = 0.1
    v_max, w_max = 0.15, 1.2  # m/s, rad/s

    wps = waypoints_padrao(mapa)
    theta0 = np.deg2rad(mapa["robo"]["theta_graus"]) if mapa["robo"] else 0.0
    real = np.array([wps[0][0], wps[0][1], theta0])

    # ---- Parâmetros do ruído (da realidade) ----
    sd_s, sd_th = 0.05, 0.04         # fração do deslocamento / ruído de giro (rad por passo base)
    vies_theta = np.deg2rad(0.15)    # viés de giro por passo (deriva da odometria)
    sd_cam_xy = args.ruido_cam       # m
    sd_cam_th = np.deg2rad(4.0)
    prob_perda = 1.0 if args.sem_camera else args.perda
    prob_outlier = args.outlier
    periodo_cam = 5                  # câmera a cada 5 passos (2 Hz)

    # ---- Matrizes do filtro (o que o filtro "acha" do ruído) ----
    R = np.diag([0.003 ** 2, 0.003 ** 2, np.deg2rad(1.0) ** 2])
    Q = np.diag([sd_cam_xy ** 2, sd_cam_xy ** 2, sd_cam_th ** 2])

    # Estado inicial do filtro propositalmente errado
    mu = real + np.array([0.08, -0.06, np.deg2rad(12)])
    mu[2] = kf.norm_angle(mu[2])
    sigma = np.diag([0.1 ** 2, 0.1 ** 2, np.deg2rad(15) ** 2])
    odo = mu.copy()  # odometria pura (dead-reckoning)

    hist = {k: [] for k in ["t", "real", "ekf", "odo", "cam", "cam_t", "dp", "outlier"]}
    alvo = 1
    t = 0.0
    passo = 0
    while alvo < len(wps) and t < 400:
        # --- controle simples (verdade) ---
        dx, dy = wps[alvo][0] - real[0], wps[alvo][1] - real[1]
        dist = np.hypot(dx, dy)
        if dist < 0.03:
            alvo += 1
            continue
        erro_ang = kf.norm_angle(np.arctan2(dy, dx) - real[2])
        w = np.clip(2.5 * erro_ang, -w_max, w_max)
        v = v_max * max(0.0, np.cos(erro_ang)) ** 2
        v = min(v, dist / dt)

        ds_true, dth_true = v * dt, w * dt

        # --- movimento real ---
        tm = real[2] + dth_true / 2
        real = real + np.array([ds_true * np.cos(tm), ds_true * np.sin(tm), dth_true])
        real[2] = kf.norm_angle(real[2])

        # --- odometria medida (ruidosa + viés) ---
        ds_m = ds_true * (1 + rng.normal(0, sd_s)) if ds_true > 0 else 0.0
        dth_m = dth_true + rng.normal(0, sd_th * 0.1) + vies_theta
        odo, _ = kf.ekf_pred(odo, np.eye(3), ds_m, dth_m, np.zeros((3, 3)))

        # --- câmera ---
        xc = yc = thc = None
        outlier = False
        if passo % periodo_cam == 0 and rng.random() > prob_perda:
            xc = real[0] + rng.normal(0, sd_cam_xy)
            yc = real[1] + rng.normal(0, sd_cam_xy)
            thc = np.rad2deg(real[2] + rng.normal(0, sd_cam_th))
            if rng.random() < prob_outlier:  # falha de detecção (ex.: sombra confundida com marcador)
                xc += rng.uniform(-0.4, 0.4); yc += rng.uniform(-0.4, 0.4)
                outlier = True

        mu, sigma = kf.ekf_step(mu, sigma, ds_m, dth_m, xc, yc, thc, mapa, R, Q)

        hist["t"].append(t)
        hist["real"].append(real.copy()); hist["ekf"].append(mu.copy()); hist["odo"].append(odo.copy())
        hist["dp"].append(np.sqrt(np.diag(sigma)))
        if xc is not None:
            hist["cam"].append([xc, yc, np.deg2rad(thc)]); hist["cam_t"].append(t); hist["outlier"].append(outlier)
        t += dt
        passo += 1

    for k in ["real", "ekf", "odo", "cam", "dp"]:
        hist[k] = np.array(hist[k])
    hist["wps"] = wps
    return hist


def metricas(h):
    real, ekf, odo = h["real"], h["ekf"], h["odo"]

    def rmse(est):
        return np.sqrt(np.mean(np.sum((est[:, :2] - real[:, :2]) ** 2, axis=1)))

    def rmse_th(est):
        return np.rad2deg(np.sqrt(np.mean(kf.norm_angle(est[:, 2] - real[:, 2]) ** 2)))

    res = {"rmse_pos_ekf": rmse(ekf), "rmse_pos_odo": rmse(odo),
           "rmse_th_ekf": rmse_th(ekf), "rmse_th_odo": rmse_th(odo),
           "erro_final_ekf": np.linalg.norm(ekf[-1, :2] - real[-1, :2]),
           "erro_final_odo": np.linalg.norm(odo[-1, :2] - real[-1, :2])}
    if len(h["cam"]):
        # Câmera pura (somente instantes com medição) - exclui outliers
        idx = (np.array(h["cam_t"]) / 0.1).round().astype(int)
        idx = idx[idx < len(real)]
        cam = h["cam"][:len(idx)]
        res["rmse_pos_cam"] = np.sqrt(np.mean(np.sum((cam[:, :2] - real[idx, :2]) ** 2, axis=1)))
        res["n_cam"] = len(idx)
        res["n_outliers"] = int(np.sum(h["outlier"]))
    # NEES (consistência): erro normalizado deveria ficar em torno de 3 (dimensão do estado)
    return res


def grafico(mapa, h, args):
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[:, :2])

    if mapa["contorno"] is not None:
        c = np.vstack([mapa["contorno"], mapa["contorno"][:1]])
        ax.plot(c[:, 0], c[:, 1], "k-", lw=2, label="Mesa (fita)")
    for i, o in enumerate(mapa["obstaculos"]):
        ax.fill(o[:, 0], o[:, 1], color="tab:red", alpha=0.4, label="Obstáculo" if i == 0 else None)
    ax.plot(h["real"][:, 0], h["real"][:, 1], "g-", lw=2.5, label="Real (verdade)")
    ax.plot(h["odo"][:, 0], h["odo"][:, 1], "-", color="tab:orange", lw=1.5, label="Odometria pura")
    ax.plot(h["ekf"][:, 0], h["ekf"][:, 1], "b-", lw=1.5, label="EKF")
    if len(h["cam"]):
        out = np.array(h["outlier"])
        ax.plot(h["cam"][~out, 0], h["cam"][~out, 1], ".", color="gray", ms=4, label="Câmera")
        if out.any():
            ax.plot(h["cam"][out, 0], h["cam"][out, 1], "x", color="m", ms=7, label="Câmera (outlier)")
    ax.plot(*h["real"][0, :2], "go", ms=9)
    ax.plot(*h["real"][-1, :2], "g*", ms=14)
    wp = np.array(h["wps"])
    ax.plot(wp[:, 0], wp[:, 1], "k:", lw=1, label="Waypoints")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title("Trajetória no mapa")

    t = h["t"]
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(t, np.linalg.norm(h["odo"][:, :2] - h["real"][:, :2], axis=1), color="tab:orange", label="Odometria")
    ax2.plot(t, np.linalg.norm(h["ekf"][:, :2] - h["real"][:, :2], axis=1), "b", label="EKF")
    ax2.plot(t, 2 * np.hypot(h["dp"][:, 0], h["dp"][:, 1]) / np.sqrt(2), "b--", alpha=0.5, label="±2σ do EKF")
    ax2.set_title("Erro de posição [m]"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(t, np.rad2deg(kf.norm_angle(h["odo"][:, 2] - h["real"][:, 2])), color="tab:orange", label="Odometria")
    ax3.plot(t, np.rad2deg(kf.norm_angle(h["ekf"][:, 2] - h["real"][:, 2])), "b", label="EKF")
    ax3.set_title("Erro de orientação [°]"); ax3.set_xlabel("t [s]"); ax3.grid(alpha=0.3); ax3.legend(fontsize=8)

    fig.tight_layout()
    saida = os.path.join(AQUI, "teste_kalman.png")
    fig.savefig(saida, dpi=110)
    print(f"[OK] Gráfico salvo em {saida}")
    if args.show:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapa", default=os.path.join(AQUI, "mapa_obstaculos.json"))
    ap.add_argument("--ruido-cam", type=float, default=0.015, help="desvio-padrão da câmera em m")
    ap.add_argument("--perda", type=float, default=0.3, help="probabilidade de a câmera não ver o robô")
    ap.add_argument("--outlier", type=float, default=0.0, help="probabilidade de medição absurda da câmera")
    ap.add_argument("--sem-camera", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--so-unitarios", action="store_true")
    args = ap.parse_args()

    ok = testes_unitarios()
    if args.so_unitarios:
        return

    mapa = carregar_mapa(args.mapa)
    print(f"\n== Simulação (mapa: {len(mapa['obstaculos'])} obstáculos) ==")
    h = simular(mapa, args)

    # A trajetória real não pode atravessar obstáculos (valida a rota fictícia)
    bat = sum(colide(p[:2], mapa) for p in h["real"])
    print(f"Passos: {len(h['t'])} | duração: {h['t'][-1]:.1f} s | passos dentro de obstáculos: {bat}")

    m = metricas(h)
    print(f"RMSE posição  -> odometria: {m['rmse_pos_odo'] * 100:6.2f} cm | EKF: {m['rmse_pos_ekf'] * 100:6.2f} cm"
          + (f" | câmera: {m['rmse_pos_cam'] * 100:.2f} cm ({m['n_cam']} medições, {m['n_outliers']} outliers)"
             if 'rmse_pos_cam' in m else ""))
    print(f"RMSE ângulo   -> odometria: {m['rmse_th_odo']:6.2f}°   | EKF: {m['rmse_th_ekf']:6.2f}°")
    print(f"Erro final    -> odometria: {m['erro_final_odo'] * 100:6.2f} cm | EKF: {m['erro_final_ekf'] * 100:6.2f} cm")
    if not args.sem_camera and args.outlier == 0:
        print("Veredito:", "EKF MELHOR que a odometria pura" if m["rmse_pos_ekf"] < m["rmse_pos_odo"]
              else "EKF NÃO melhorou -> revisar R/Q")
    grafico(mapa, h, args)


if __name__ == "__main__":
    main()
