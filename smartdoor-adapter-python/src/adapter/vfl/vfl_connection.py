# vfl_connection.py
"""
Bridges the MBT handler ⇆ your vertical-FL Flower simulation.
"""

from __future__ import annotations
import threading, queue, logging
from typing import Dict, List, Tuple
from multiprocessing import Process

import flwr as fl
from flwr.simulation import run_simulation
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters

# ── your existing code ───────────────────────────────────────────────────
from vertical_fl.strategy import Strategy                  # ← as-is
from vertical_fl.client_app import app as client_app       # ← ClientApp instance
from vertical_fl.task import process_dataset               # ← to get labels

# ─────────────────────────────────────────────────────────────────────────
class ReportingStrategy(Strategy):
    """Wraps your Strategy so we can emit a callback after every round."""

    def __init__(self, report_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._report_fn = report_fn

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[str | ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Parameters | None, Dict[str, float]]:
        params, metrics = super().aggregate_fit(rnd, results, failures)
        # Notify the handler
        self._report_fn(rnd, metrics or {})
        return params, metrics


class VflConnection:
    """
    Runs Flower in a background thread and translates simple text
    commands to the simulation.

    IN:   "START:<rounds>"  | "RESET" | "STOP"
    OUT:  "RESET_PERFORMED" | "ROUND_DONE:<r>:<acc>" | "TRAINING_DONE:<acc>"
    """

    def __init__(self, handler) -> None:
        self._handler = handler
        self._cmd_q: queue.Queue[str] = queue.Queue()
        self._alive = threading.Event()
        self.prcess: Process

    # ––––– public API (used by Handler) ––––––––––––––––––––––––––––––––
    def connect(self) -> None:
        self._alive.set()
        threading.Thread(target=self._loop, daemon=True).start()
        self.send("RESET")
        logging.info("VFL connection established, waiting for commands...")

    def send(self, cmd: str) -> None:
        self._cmd_q.put(cmd)

    def stop(self) -> None:
        self._cmd_q.put("STOP")

    # ––––– internal worker loop ––––––––––––––––––––––––––––––––––––––––
    def _loop(self) -> None:
        while self._alive.is_set():
            cmd = self._cmd_q.get()
            if cmd == "STOP":
                self._alive.clear()
                break
            if cmd == "RESET":
                self._handler.send_message_to_amp("RESET_PERFORMED")
            elif cmd.startswith("START:"):
                rounds = int(cmd.split(":", 1)[1])
                self._run_flower(rounds)
            else:
                self._handler.send_message_to_amp(f"ERROR:UnknownCmd:{cmd}")

    # ––––– one simulation run ––––––––––––––––––––––––––––––––––––––––––
    def _run_flower(self, rounds: int) -> None:
        latest_acc = float | None
        # Build ReportingStrategy (wraps your Strategy)
        labels, _ = process_dataset()
        labels = labels["Survived"].tolist()

        def _on_round(rnd: int, metrics: Dict[str, float]) -> None:
            nonlocal latest_acc
            latest_acc = metrics.get("accuracy", -1.0)
            self._handler.send_message_to_amp(f"ROUND_DONE:{rnd}:{latest_acc:.4f}")

        strategy = ReportingStrategy(report_fn=_on_round, labels=labels)

        server_components = ServerAppComponents(
            strategy=strategy, config=ServerConfig(num_rounds=rounds)
        )
        server_app = ServerApp(server_fn=lambda _: server_components)

        self._handler.send_message_to_amp("TRAINING_STARTED")
        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=3,
            # verbose_logging=True, # Uncomment for debug output
        )

        self._handler.send_message_to_amp(f"TRAINING_DONE:{latest_acc:.4f}")
