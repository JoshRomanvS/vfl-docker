# vfl_connection.py
from multiprocessing import Process
import queue, threading
import flwr as fl
from flwr.simulation import run_simulation
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from vertical_fl.client_app import app as client_app
from vertical_fl.strategy import Strategy

class VflConnection:
    def __init__(self, handler):
        self._handler = handler
        self._cmd_q: queue.Queue[str] = queue.Queue()
        self._thr:    threading.Thread | None = None
        self._proc:   Process | None = None         # ✱ current worker proc

    # ───────────────── public API ──────────────────────────────────────
    def connect(self):
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def send(self, cmd: str):
        self._cmd_q.put(cmd)

    def stop(self):
        self._cmd_q.put("STOP")
        if self._thr:
            self._thr.join()

    # ───────────────── worker thread ───────────────────────────────────
    def _loop(self):
        while True:
            cmd = self._cmd_q.get()
            if cmd == "STOP":
                self._kill_proc()
                self._handler.send_message_to_amp("STOPPED")
                break
            if cmd == "RESET":
                self._kill_proc()
                self._handler.send_message_to_amp("RESET_PERFORMED")
            elif cmd.startswith("START:"):
                rounds = int(cmd.split(":", 1)[1])
                self._kill_proc()                     # safety: none running
                self._proc = Process(
                    target=_worker,
                    args=(rounds, self._handler),
                    daemon=True,
                )
                self._proc.start()

    # ───────────────── helpers ─────────────────────────────────────────
    def _kill_proc(self):
        if self._proc and self._proc.is_alive():
            self._proc.terminate()   # hard stop
            self._proc.join()
        self._proc = None

# ───────────────────────────────────────────────────────────────────────
def _worker(rounds: int, handler):
    """Runs in the child process."""
    latest_acc: float | None = None

    def _on_round(rnd: int, metrics: dict):
        nonlocal latest_acc
        latest_acc = metrics.get("accuracy")
        handler.send_message_to_amp(f"ROUND_DONE:{rnd}:{latest_acc:.4f}")

    strategy = Strategy()
    strategy.aggregate_fit = _wrap_aggregate_fit(strategy.aggregate_fit, _on_round)

    server_app = ServerApp(
        server_fn=lambda _: ServerAppComponents(
            strategy=strategy, config=ServerConfig(num_rounds=rounds)
        )
    )

    run_simulation(                     # blocking, but isolated in proc
        server_app=server_app,
        client_app=client_app,
        num_supernodes=3,
    )

    handler.send_message_to_amp(f"TRAINING_DONE:{latest_acc:.4f}")

# Wrap the original aggregate_fit so we don’t touch strategy.py
def _wrap_aggregate_fit(original_aggregate_fit, cb):
    def wrapper(round, results, failures):
        params, metrics = original_aggregate_fit(round, results, failures)
        cb(round, metrics or {})
        return params, metrics
    return wrapper
