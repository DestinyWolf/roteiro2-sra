#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import re
import math
import argparse
import threading
import queue
from collections import deque
from dataclasses import dataclass
from typing import Optional

import serial
import serial.tools.list_ports


# ---- Defaults (adapte ao seu adaptador RF) ------------------------------
DEFAULT_PORT = os.environ.get("RF_PORT", "COM14")
DEFAULT_BAUD = int(os.environ.get("RF_BAUD", "57600"))
DEFAULT_TMO  = float(os.environ.get("RF_TIMEOUT", "2.0"))
# -------------------------------------------------------------------------

STATE_NAMES = {0: "IDLE", 1: "GIRAR", 2: "ANDAR", 3: "DONE"}


# ---------------------------------------------------------------------------
# Parser de odometria <O,x,y,th,v,w,t_ms>
# ---------------------------------------------------------------------------
RX_ODOM_RE = re.compile(
    r"<O,"
    r"([-+0-9.]+),([-+0-9.]+),([-+0-9.]+),"   # x, y, th_deg
    r"([-+0-9.]+),([-+0-9.]+),"               # v, w
    r"(\d+)>"                                 # t_ms
)


@dataclass
class Odom:
    x: float
    y: float
    th_deg: float
    v: float
    w: float
    t_ms: int


# ---------------------------------------------------------------------------
# Portas seriais
# ---------------------------------------------------------------------------
def list_ports():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("Nenhuma porta serial encontrada.")
        return
    print("Portas disponíveis:")
    for p in ports:
        print(f"  {p.device:20s}  {p.description}")


def autodetect_port(prefer=DEFAULT_PORT):
    ports = list(serial.tools.list_ports.comports())
    names = [p.device for p in ports]
    if prefer in names:
        return prefer
    for p in ports:
        desc = (p.description or "").upper()
        if "USB" in desc or "CH340" in desc:
            return p.device
    return names[0] if names else None


# ---------------------------------------------------------------------------
# Conexão
# ---------------------------------------------------------------------------
def connect(port=DEFAULT_PORT, baud=DEFAULT_BAUD, timeout=DEFAULT_TMO):
    ser = serial.Serial()
    ser.port      = port
    ser.baudrate  = baud
    ser.bytesize  = serial.EIGHTBITS
    ser.parity    = serial.PARITY_NONE
    ser.stopbits  = serial.STOPBITS_ONE
    ser.timeout   = 0.05
    ser.write_timeout = 2.0
    ser.dtr       = False
    ser.rts       = False
    ser.open()

    rx_q = queue.Queue()
    running = {"v": True}

    def rx_loop():
        while running["v"]:
            try:
                line = ser.readline()
            except serial.SerialException:
                running["v"] = False
                break
            if not line:
                continue
            s = line.decode("ascii", errors="replace").strip()
            if s:
                rx_q.put(s)

    t = threading.Thread(target=rx_loop, name="rf-rx", daemon=True)
    t.start()

    time.sleep(0.3)
    while not rx_q.empty():
        try: rx_q.get_nowait()
        except queue.Empty: break

    return ser, rx_q, running, t


def close(ser, running, t):
    running["v"] = False
    if t.is_alive():
        t.join(timeout=1.0)
    try:
        ser.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Envio / leitura genéricos
# ---------------------------------------------------------------------------
def send(ser, cmd):
    ser.write(f"<{cmd}>\r\n".encode("ascii"))
    ser.flush()


def recv_body(rx_q, timeout=DEFAULT_TMO):
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            line = rx_q.get(timeout=remaining)
        except queue.Empty:
            return None
        m = re.search(r"<([^>]*)>", line)
        if not m:
            continue
        body = m.group(1)
        if body == "ERR":
            return "<ERR>"
        return body


def cmd_and_wait(ser, rx_q, cmd, timeout=DEFAULT_TMO):
    send(ser, cmd)
    body = recv_body(rx_q, timeout)
    return None if body is None else f"<{body}>"


# ---------------------------------------------------------------------------
# Comandos de baixo nível
# ---------------------------------------------------------------------------
def send_v(ser, v_lin, v_ang):
    """<V,v,w> — referência de velocidade (m/s, rad/s). Não espera ack."""
    try:
        ser.write(f"<V,{v_lin:.4f},{v_ang:.4f}>\r\n".encode("ascii"))
        ser.flush()
    except serial.SerialException:
        pass


def stop(ser, rx_q):
    return cmd_and_wait(ser, rx_q, "P")


def set_pose(ser, rx_q, x, y, th):
    return cmd_and_wait(ser, rx_q, f"Z,{x:.4f},{y:.4f},{th:.2f}")


def drain_odom(rx_q):
    while True:
        try:
            rx_q.get_nowait()
        except queue.Empty:
            return


# ---------------------------------------------------------------------------
# Handshake
# ---------------------------------------------------------------------------
def handshake(ser, rx_q, retries=10, per_try=0.3):
    for _ in range(retries):
        send(ser, "E")
        if recv_body(rx_q, per_try) is not None:
            while not rx_q.empty():
                try: rx_q.get_nowait()
                except queue.Empty: break
            return True
    return False


# ---------------------------------------------------------------------------
# Telemetria / estatísticas
# ---------------------------------------------------------------------------
def get_telemetry(ser, rx_q, timeout=DEFAULT_TMO):
    send(ser, "T")
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            line = rx_q.get(timeout=remaining)
        except queue.Empty:
            return None
        m = RX_ODOM_RE.match(line)
        if not m:
            continue
        x, y, th, v, w, t = m.groups()
        return {
            "x":      float(x),
            "y":      float(y),
            "th_deg": float(th),
            "v":      float(v),
            "w":      float(w),
            "w_deg":  float(w) * 180.0 / math.pi,
            "t_ms":   int(t),
            "wp":     -1,
            "state":  0,
        }


def get_stats(ser, rx_q, timeout=DEFAULT_TMO):
    resp = cmd_and_wait(ser, rx_q, "E", timeout)
    if not resp or not resp.startswith("<E,"):
        return None
    parts = resp[3:-1].split(",")
    if len(parts) not in (3, 4):
        return None
    try:
        d = {"errors": int(parts[0]), "dropped": int(parts[1])}
        if len(parts) == 4:
            d["isr"]    = int(parts[2])
            d["rx_idx"] = int(parts[3])
        else:
            d["rx_idx"] = int(parts[2])
        return d
    except ValueError:
        return None


def format_telemetry(t):
    if t is None:
        return "(sem telemetria)"
    return (f"x={t['x']:+.3f}  y={t['y']:+.3f}  "
            f"th={t['th_deg']:+7.2f}°  "
            f"v={t['v']:+.3f} m/s  w={t['w_deg']:+6.2f} °/s")


def print_telemetry(t):
    print(format_telemetry(t))


# ---------------------------------------------------------------------------
# ===== CONTROLADOR DE POSIÇÃO EM ALTO NÍVEL ================================
# ---------------------------------------------------------------------------
def wrap_pi(a):
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


@dataclass
class HLCConfig:
    eps_pos: float = 0.010
    eps_th_deg: float = 2.0
    d_ent_deg: float = 15.0
    d_sai_deg: float = 1.0
    k_lin: float = 1.5
    k_ang: float = 2.0
    k_ang_final: float = 3.0
    v_max: float = 0.20
    v_min: float = 0.02
    w_max: float = math.radians(60)
    w_max_final: float = math.radians(45)
    loop_dt: float = 0.010
    odom_watchdog_s: float = 0.5
    cmd_watchdog_s: float  = 0.5


class PositionController:
    """Mesma lógica do Nav_Step do firmware antigo, agora em Python."""

    def __init__(self, cfg: HLCConfig):
        self.cfg = cfg
        self.em_giro = False
        self.wp = None

    def set_target(self, wp):
        """Aceita None (sem alvo) ou tupla (x, y, th_deg)."""
        if wp is None:
            self.wp = None
            self.em_giro = False
        else:
            x, y, th_deg = wp
            self.wp = (x, y, th_deg)
            self.em_giro = False

    def step(self, odom: Odom):
        cfg = self.cfg
        if self.wp is None:
            return 0.0, 0.0, True

        wx, wy, wth_deg = self.wp
        th = math.radians(odom.th_deg)

        dx = wx - odom.x
        dy = wy - odom.y
        dl = math.hypot(dx, dy)

        # Fase 1 — longe do waypoint
        if dl >= cfg.eps_pos:
            th_d = math.atan2(dy, dx)
            dth = wrap_pi(th_d - th)

            if (not self.em_giro) and abs(dth) > math.radians(cfg.d_ent_deg):
                self.em_giro = True
            if self.em_giro and abs(dth) < math.radians(cfg.d_sai_deg):
                self.em_giro = False

            if self.em_giro:
                v = 0.0
                w = cfg.k_ang * dth
            else:
                v = cfg.k_lin * dl
                v = max(min(v, cfg.v_max), cfg.v_min)
                w = cfg.k_ang * dth

            v = max(min(v, cfg.v_max), -cfg.v_max)
            w = max(min(w, cfg.w_max), -cfg.w_max)
            return v, w, False

        # Fase 2 — alinhar orientação final
        dth_f = wrap_pi(math.radians(wth_deg) - th)
        if abs(dth_f) > math.radians(cfg.eps_th_deg):
            w = cfg.k_ang_final * dth_f
            w = max(min(w, cfg.w_max_final), -cfg.w_max_final)
            return 0.0, w, False

        return 0.0, 0.0, True


class RobotRunner:
    """Loop de controle a 100 Hz: lê <O,...>, calcula (v,w), envia <V,v,w>."""

    def __init__(self, ser, rx_q, cfg: HLCConfig = None):
        self.ser = ser
        self.rx_q = rx_q
        self.cfg = cfg or HLCConfig()
        self.ctrl = PositionController(self.cfg)
        self.queue = deque()
        self.current = None
        self.last_odom_t = time.time()
        self.running = False

    def enqueue(self, pts):
        for (x, y, th) in pts:
            self.queue.append((x, y, th))

    def clear_queue(self):
        self.queue.clear()
        self.current = None
        self.ctrl.set_target(None)
        send_v(self.ser, 0.0, 0.0)

    def read_odom_nowait(self) -> Optional[Odom]:
        """Pega a odometria mais recente do buffer (descarta as antigas)."""
        latest = None
        while True:
            try:
                line = self.rx_q.get_nowait()
            except queue.Empty:
                break
            m = RX_ODOM_RE.match(line)
            if m:
                x, y, th, v, w, t = m.groups()
                latest = Odom(float(x), float(y), float(th),
                              float(v), float(w), int(t))
        return latest

    def run_once(self) -> Optional[Odom]:
        """Um passo do loop. Retorna odometria usada (ou None)."""
        odom = self.read_odom_nowait()
        if odom is not None:
            self.last_odom_t = time.time()

        # watchdog de odometria
        if (time.time() - self.last_odom_t) > self.cfg.odom_watchdog_s:
            send_v(self.ser, 0.0, 0.0)
            return None

        # pega próximo waypoint, se preciso
        if self.current is None:
            if self.queue:
                self.current = self.queue.popleft()
                self.ctrl.set_target(self.current)
                print(f"[HLC] -> waypoint ({self.current[0]:.3f}, "
                      f"{self.current[1]:.3f}, {self.current[2]:.1f}°)")
            else:
                send_v(self.ser, 0.0, 0.0)
                return odom

        if odom is None:
            return None

        v, w, done = self.ctrl.step(odom)

        if done:
            print(f"[HLC] waypoint atingido: ({self.current[0]:.3f}, "
                  f"{self.current[1]:.3f}, {self.current[2]:.1f}°)")
            send_v(self.ser, 0.0, 0.0)          # para ANTES de limpar
            self.current = None
            self.ctrl.set_target(None)
            if not self.queue:
                print("[HLC] fila vazia — robô parado.")
            return odom

        send_v(self.ser, v, w)
        return odom

    def run_loop(self, timeout_s=120.0, verbose=True):
        """Executa até terminar fila, timeout, Ctrl+C ou stop externo."""
        self.running = True
        t0 = time.time()
        next_tick = time.perf_counter()
        started = False                     # só True após processar 1º wp
        try:
            while self.running and (time.time() - t0) < timeout_s:
                now = time.perf_counter()
                if now < next_tick:
                    time.sleep(next_tick - now)
                next_tick += self.cfg.loop_dt

                self.run_once()
                if self.current is not None:
                    started = True

                if started and self.current is None and not self.queue:
                    return True
        except KeyboardInterrupt:
            print("\n[HLC] interrompido.")
            return False
        finally:
            self.running = False
            send_v(self.ser, 0.0, 0.0)
        return False


# ---------------------------------------------------------------------------
# Missão (CSV ou lista)
# ---------------------------------------------------------------------------
def run_mission(ser, rx_q, pts, poll_s=0.5, timeout_s=120.0):
    print(f"[1/3] Reset de pose para (0,0,0)...")
    set_pose(ser, rx_q, 0.0, 0.0, 0.0)

    print(f"[2/3] Enfileirando {len(pts)} waypoint(s)...")
    for i, (x, y, th) in enumerate(pts):
        print(f"      [{i}] ({x:.3f}, {y:.3f}, {th:.1f}°)")

    print(f"[3/3] Iniciando controle a 100 Hz (timeout {timeout_s:.0f}s)...")
    runner = RobotRunner(ser, rx_q)
    runner.enqueue(pts)
    ok = runner.run_loop(timeout_s=timeout_s)
    return ok, ("missão concluída" if ok else "timeout/interrompida")


def go(ser, rx_q, active_runner, x, y, th):
    runner = RobotRunner(ser, rx_q=rx_q)
    active_runner['r'] = runner
    runner.enqueue([x, y, th])
    result = False
    try:
        result = runner.run_loop(timeout_s=60.0)
    finally:
        active_runner['r'] = None

    if result:
        return True, "sucesso"
    return False, "falha"

