import logging, threading, queue, time, flwr as fl
from ../../../quickstart-docker.vertical_fl import server_app, client_app


class VflConnection:
    def __init__(self, handler):
        self.handler = handler
        self.cmd_q   = queue.Queue()
        self.thread  = None
        self.alive   = threading.Event()

    # ───── External API used by Handler ──────────────────────────────────
    def connect(self):
        self.alive.set()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logging.info("VFL driver thread started")
        # Immediately acknowledge so AMP can send next stimulus
        self.handler.send_message_to_amp("RESET_PERFORMED")

    def send(self, cmd: str):
        logging.debug("Injecting cmd into VFL driver: %s", cmd)
        self.cmd_q.put(cmd)

    def stop(self):
        self.cmd_q.put("STOP")
        self.thread.join(timeout=5)

    # ───── Internal worker loop ─────────────────────────────────────────
    def _run(self):
        try:
            while self.alive.is_set():
                cmd = self.cmd_q.get()
                if cmd == "STOP":
                    self.alive.clear()
                    break
                if cmd.startswith("START:"):
                    rounds = int(cmd.split(":")[1])
                    self._train(rounds)
                elif cmd == "RESET":
                    # Nothing to reset yet; future-proof
                    self.handler.send_message_to_amp("RESET_PERFORMED")
                else:
                    self.handler.send_message_to_amp(f"ERROR:UnknownCmd:{cmd}")
        except Exception as exc:
            logging.exception("VFL driver crashed")
            self.handler.send_message_to_amp(f"ERROR:{exc}")

    # ───── One training run ─────────────────────────────────────────────
    def _train(self, rounds: int):
        from flwr.runner import run_simulation  # Local-mode helper
        # Every round callback returns (round, metrics)
        def _after_round(rnd, metrics):
            acc = metrics.get("accuracy", -1)
            self.handler.send_message_to_amp(f"ROUND_DONE:{rnd}:{acc:.4f}")

        # Spin up the server + clients in-process (no network ports)
        history = run_simulation(
            client_fn=client_app.client_fn,
            num_clients=3,
            server_fn=server_app.server_fn,
            config=fl.server.ServerConfig(num_rounds=rounds),
            after_round=_after_round,
        )

        final_acc = history.metrics_distributed["accuracy"][-1][1]
        self.handler.send_message_to_amp(f"TRAINING_DONE:{final_acc:.4f}")
