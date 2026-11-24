#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controle remoto do Minibalance via RF (nRF24L01) — porta COM do adaptador.

Arquitetura:
  PC ──USB──> [Adaptador RF: Arduino/ESP32 + nRF24L01] ──2.4GHz──> [nRF24L01 no STM32]

Protocolo (firmware de baixo nível):
  PC → MCU:
    <V,v_lin,v_ang>     referência de velocidade  (m/s, rad/s)  [PRINCIPAL]
    <P>                 parada imediata
    <Z,x,y,th>          redefine pose (reset de odometria)
    <T>                 pede odometria sob demanda
    <E>                 estatísticas RX
  MCU → PC:
    <O,x,y,th,v,w,t_ms> odometria (~100 Hz)
    <BOOT>              no boot
    <OK> / <ERR>        ack dos comandos

Este script implementa TODA a estratégia de posição em alto nível (Python),
enviando <V,v,w> ao robô a 100 Hz.
"""

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


# ---------------------------------------------------------------------------
# CLI interativo
# ---------------------------------------------------------------------------
HELP = """
Comandos:
  add X Y TH              adiciona waypoint ao final (da fila local)
  set X1 Y1 T1 [...]      SUBSTITUI a fila local
  app X1 Y1 T1 [...]      ADICIONA à fila local
  list                    lista a fila local
  del IDX                 remove waypoint IDX da fila local
  sub IDX X Y TH          substitui waypoint IDX da fila local
  go X Y TH               waypoint único (executa imediatamente)
  pose X Y TH             redefine pose do robô (<Z>)
  start                   inicia execução da fila local
  stop                    parada de emergência (<P>)
  clear                   ZERA a fila local e para
  run                     start + monitora até DONE/timeout
  tm                      telemetria: uma amostra
  tm monitor              monitora telemetria a 2 Hz (Ctrl+C para sair)
  st                      estatísticas de RX (<E>)
  raw <texto>             envia texto cru dentro de <...>
  h                       esta ajuda
  q                       sai
"""


def parse_floats(args):
    toks = []
    for a in args:
        toks.extend(a.replace(",", " ").split())
    vals = [float(t) for t in toks]
    if not vals or len(vals) % 3 != 0:
        raise ValueError("Precisa ser múltiplo de 3 (x y th)")
    return [(vals[i], vals[i + 1], vals[i + 2]) for i in range(0, len(vals), 3)]


def interactive(ser, rx_q):
    queue_local = deque()
    monitor_stop = {"v": False}
    active_runner = {"r": None}          # referência compartilhada p/ stop

    print(HELP)
    while True:
        try:
            line = input("rf> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        try:
            if cmd in ("q", "quit", "exit"):
                break

            elif cmd in ("h", "help"):
                print(HELP)

            elif cmd == "raw":
                txt = line[len("raw"):].strip()
                print(cmd_and_wait(ser, rx_q, txt))

            elif cmd == "add":
                pts = parse_floats(args)
                if len(pts) != 1:
                    print("add espera X Y TH"); continue
                queue_local.append(pts[0])
                print(f"OK ({len(queue_local)} na fila)")

            elif cmd == "set":
                pts = parse_floats(args)
                queue_local.clear()
                queue_local.extend(pts)
                print(f"OK ({len(queue_local)} na fila)")

            elif cmd == "app":
                pts = parse_floats(args)
                queue_local.extend(pts)
                print(f"OK ({len(queue_local)} na fila)")

            elif cmd == "list":
                if not queue_local:
                    print("(fila vazia)")
                else:
                    print(f"{len(queue_local)} waypoint(s):")
                    for i, (x, y, th) in enumerate(queue_local):
                        print(f"  [{i:2d}] x={x:.3f}  y={y:.3f}  th={th:.1f}")

            elif cmd == "del":
                idx = int(args[0])
                if 0 <= idx < len(queue_local):
                    del queue_local[idx]
                    print("OK")
                else:
                    print("índice inválido")

            elif cmd == "sub":
                idx = int(args[0])
                pts = parse_floats(args[1:])
                if len(pts) != 1:
                    print("sub espera IDX X Y TH"); continue
                if 0 <= idx < len(queue_local):
                    queue_local[idx] = pts[0]
                    print("OK")
                else:
                    print("índice inválido")

            elif cmd == "go":
                pts = parse_floats(args)
                if len(pts) != 1:
                    print("go espera X Y TH"); continue
                x, y, th = pts[0]
                print(f"[HLC] go ({x:.3f}, {y:.3f}, {th:.1f}°)")
                runner = RobotRunner(ser, rx_q)
                active_runner["r"] = runner
                runner.enqueue([(x, y, th)])
                try:
                    runner.run_loop(timeout_s=60.0)
                finally:
                    active_runner["r"] = None

            elif cmd == "pose":
                pts = parse_floats(args)
                if len(pts) != 1:
                    print("pose espera X Y TH"); continue
                x, y, th = pts[0]
                print(set_pose(ser, rx_q, x, y, th))

            elif cmd == "start":
                if not queue_local:
                    print("Fila vazia. Use 'add'/'set' antes."); continue
                runner = RobotRunner(ser, rx_q)
                active_runner["r"] = runner
                runner.enqueue(list(queue_local))
                queue_local.clear()
                print("[HLC] iniciando controle...")
                try:
                    runner.run_loop(timeout_s=120.0)
                finally:
                    active_runner["r"] = None

            elif cmd == "stop":
                monitor_stop["v"] = True
                r = active_runner["r"]
                if r is not None:
                    r.running = False
                print(stop(ser, rx_q))

            elif cmd == "clear":
                queue_local.clear()
                r = active_runner["r"]
                if r is not None:
                    r.clear_queue()
                else:
                    send_v(ser, 0.0, 0.0)
                print(stop(ser, rx_q))

            elif cmd == "run":
                if not queue_local:
                    print("Fila vazia. Use 'set'/'add' antes de 'run'.")
                    continue
                runner = RobotRunner(ser, rx_q)
                active_runner["r"] = runner
                runner.enqueue(list(queue_local))
                queue_local.clear()
                print("[HLC] iniciando controle (run)...")
                try:
                    runner.run_loop(timeout_s=120.0)
                finally:
                    active_runner["r"] = None

            elif cmd == "tm":
                if args and args[0].lower() == "monitor":
                    print("Monitorando telemetria a 2 Hz (Ctrl+C para sair)...")
                    monitor_stop["v"] = False
                    try:
                        while not monitor_stop["v"]:
                            print_telemetry(get_telemetry(ser, rx_q))
                            time.sleep(0.5)
                    except KeyboardInterrupt:
                        print("\n(monitor encerrado)")
                else:
                    print_telemetry(get_telemetry(ser, rx_q))

            elif cmd == "st":
                s = get_stats(ser, rx_q)
                print(s if s else "(sem resposta)")

            else:
                print(f"Comando desconhecido: {cmd}. Digite 'h'.")

        except ValueError as e:
            print(f"Erro de argumento: {e}")
        except IndexError:
            print("Faltam argumentos.")
        except serial.SerialException as e:
            print(f"Erro serial: {e}")
            break
        except KeyboardInterrupt:
            print("\n(interrompido)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=f"Controle do Minibalance via RF — padrão {DEFAULT_PORT} @ {DEFAULT_BAUD}")
    ap.add_argument("--port", default=DEFAULT_PORT,
                    help=f"porta serial do adaptador RF (default: {DEFAULT_PORT})")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD,
                    help=f"baudrate (default: {DEFAULT_BAUD})")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TMO,
                    help=f"timeout por comando em segundos (default: {DEFAULT_TMO})")
    ap.add_argument("--list", action="store_true", help="lista portas e sai")
    ap.add_argument("--autodetect", action="store_true",
                    help="tenta achar a porta do adaptador RF automaticamente")
    ap.add_argument("--send", help="envia um comando (sem <>) e sai")
    ap.add_argument("--tm", action="store_true", help="uma amostra de telemetria e sai")
    ap.add_argument("--stats", action="store_true", help="estatísticas <E> e sai")
    ap.add_argument("--csv", help="CSV 'x,y,th' por linha; executa missão")
    ap.add_argument("--monitor", action="store_true",
                    help="monitora telemetria continuamente (Ctrl+C)")
    ap.add_argument("--reset-pose", default="0,0,0",
                    help="pose inicial antes da missão (x,y,th). Padrão: 0,0,0")
    args = ap.parse_args()

    if args.list:
        list_ports()
        return 0

    port = args.port
    if args.autodetect:
        detected = autodetect_port(port)
        if detected:
            port = detected
            print(f"[autodetect] usando {port}")
        else:
            print("Nenhuma porta encontrada."); return 1

    try:
        ser, rx_q, running, t = connect(port, args.baud, args.timeout)
    except serial.SerialException as e:
        print(f"Não consegui abrir {port} @ {args.baud}: {e}")
        print("Verifique: adaptador RF conectado, Monitor Serial fechado, porta correta.")
        return 1

    print(f"Conectado em {port} @ {args.baud} baud.")

    try:
        if not handshake(ser, rx_q):
            print("[AVISO] Handshake <E> não respondeu. Tentando mesmo assim...")

        if args.send:
            print(cmd_and_wait(ser, rx_q, args.send))

        elif args.tm:
            print_telemetry(get_telemetry(ser, rx_q))

        elif args.stats:
            print(get_stats(ser, rx_q))

        elif args.monitor:
            print("Monitorando a 2 Hz (Ctrl+C para sair)...")
            try:
                while True:
                    print_telemetry(get_telemetry(ser, rx_q))
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\n(monitor encerrado)")

        elif args.csv:
            pts = []
            with open(args.csv, "r", encoding="utf-8") as f:
                for ln, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    toks = line.replace(";", ",").split(",")
                    if len(toks) < 3:
                        print(f"[{ln}] linha inválida: {line}"); continue
                    pts.append((float(toks[0]), float(toks[1]), float(toks[2])))
            if not pts:
                print("CSV sem waypoints válidos."); return 1

            try:
                rx, ry, rth = (float(v) for v in args.reset_pose.split(","))
            except ValueError:
                print("ERRO: --reset-pose deve ser 'x,y,th'", file=sys.stderr)
                return 2
            set_pose(ser, rx_q, rx, ry, rth)

            ok, msg = run_mission(ser, rx_q, pts)
            print(f"Resultado: {msg}")

        else:
            interactive(ser, rx_q)

    finally:
        close(ser, running, t)
        print("Desconectado.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
