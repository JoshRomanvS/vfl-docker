"""
Bridges the MBT handler ⇆ your vertical-FL Flower simulation.
"""

from __future__ import annotations
import logging
import queue
import threading
from enum import Enum
from typing import Dict, List, Tuple, Optional, Callable
import random

import flwr as fl
from flwr.common import FitRes, Parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.simulation import run_simulation

# ── your existing modules ───────────────────────────────────────────────
from vertical_fl.client_app import app as client_app, CHECKPOINT_DIR as CLIENT_CKPT_DIR
from vertical_fl.strategy import Strategy, CHECKPOINT_DIR as SERVER_CKPT_DIR
from vertical_fl.task import process_dataset             # to get labels
# -----------------------------------------------------------------------

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Enumeration of message types for better type safety."""
    TRAINING_STARTED = "TRAINING_STARTED"
    ROUND_DONE = "ROUND_DONE"
    TRAINING_DONE = "TRAINING_DONE"
    RESET_PERFORMED = "RESET_PERFORMED"
    HARD_RESET_PERFORMED = "HARD_RESET_PERFORMED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class Command(Enum):
    """Enumeration of command types."""
    START = "START"
    RESET = "RESET"
    RESET_VFL = "RESET_VFL"  # Reset current VFL run (same as RESET)
    STOP = "STOP"  # Stop running SUT (same as RESET)
    SHUTDOWN = "SHUTDOWN"  # Shutdown the entire connection


class StopSimulation(Exception):
    """Raised inside the strategy to abort the run after the current round."""


class ReportingStrategy(Strategy):
    """Wraps your Strategy to:
       1) emit metrics after every round and
       2) honour a stop event set by the adapter.
    """

    def __init__(
        self,
        report_fn: Callable[[int, Dict[str, float]], None],
        stop_evt: threading.Event,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._report_fn = report_fn
        self._stop_evt = stop_evt

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[str | ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Parameters | None, Dict[str, float]]:
        """Aggregate fit results and report metrics."""
        params, metrics = super().aggregate_fit(rnd, results, failures)

        # Report metrics after aggregation
        self._report_fn(rnd, metrics or {})

        # Check if stop was requested
        if self._stop_evt.is_set():
            logger.info(f"Stop requested after round {rnd}")
            raise StopSimulation

        return params, metrics


class VflConnection:
    """Adapter-side controller for MBT ⇆ Flower VFL simulation.

    Commands IN:
        START:<n>  - Start training with n rounds
        RESET      - Reset/stop current training
        STOP       - Stop current training (same as RESET)
        SHUTDOWN   - Shutdown adapter completely

    Messages OUT:
        TRAINING_STARTED
        ROUND_DONE:<round>:<accuracy|na>
        TRAINING_DONE[:<accuracy>]
        RESET_PERFORMED
        ERROR:<reason>
    """

    def __init__(self, handler):
        self._handler = handler
        self._cmd_q: queue.Queue[str] = queue.Queue()
        self._main_thread: Optional[threading.Thread] = None
        self._sim_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running = False
        self._lock = threading.Lock()

    def connect(self) -> None:
        """Start the connection and begin listening for commands."""
        with self._lock:
            if self._is_running:
                logger.warning("VFL connection already running")
                return

            self._is_running = True
            self._main_thread = threading.Thread(
                target=self._main_loop,
                daemon=True,
                name="VFL-MainLoop"
            )
            self._main_thread.start()

        logger.info("VFL connection ready; waiting for commands")
        self.send_command(Command.RESET.value)

    def send_command(self, cmd: str) -> None:
        """Send a command to the connection."""
        if not self._is_running:
            logger.warning(f"Cannot send command {cmd}: connection not running")
            return
        self._cmd_q.put(cmd)

    def stop(self) -> None:
        """Stop the connection gracefully."""
        with self._lock:
            if not self._is_running:
                return

        self.send_command(Command.SHUTDOWN.value)

        if self._main_thread:
            self._main_thread.join(timeout=5.0)
            if self._main_thread.is_alive():
                logger.warning("Main thread did not terminate within timeout")

    def clear_all_checkpoints(self) -> None:
        """Delete every *.pth file produced by the current VFL run."""
        for dir_path in (SERVER_CKPT_DIR, CLIENT_CKPT_DIR):
            for checkpoint in dir_path.glob("*.pth"):
                try:
                    checkpoint.unlink()
                except OSError as err:
                    logger.warning(f"Could not delete {checkpoint}: {err}")
        logger.info("Cleared all checkpoints")

    def _send_message(self, msg_type: MessageType, *args) -> None:
        """Send a message to the MBT handler."""
        if args:
            message = f"{msg_type.value}:{':'.join(map(str, args))}"
        else:
            message = msg_type.value
        self._handler.send_message_to_amp(message)

    def _send_error(self, reason: str) -> None:
        """Send an error message."""
        self._send_message(MessageType.ERROR, reason)

    def _main_loop(self) -> None:
        """Main command processing loop."""
        logger.info("Starting VFL main loop")

        try:
            while self._is_running:
                try:
                    # Use timeout to allow periodic checks of _is_running
                    cmd = self._cmd_q.get(timeout=1.0)
                except queue.Empty:
                    continue

                if not self._is_running:
                    break

                self._process_command(cmd)

        except Exception as e:
            logger.exception("Unexpected error in main loop")
            self._send_error(f"MainLoopError:{str(e)}")
        finally:
            self._graceful_stop_simulation()
            logger.info("VFL main loop terminated")

    def _process_command(self, cmd: str) -> None:
        """Process a single command."""
        try:
            if cmd == Command.SHUTDOWN.value:
                self._is_running = False
                return

            if cmd in (Command.STOP.value):
                self._graceful_stop_simulation()
                self._send_message(MessageType.STOPPED)
                return

            if cmd in (Command.RESET.value, Command.RESET_VFL.value):
                # print(F"command received: {cmd}")
                self._graceful_stop_simulation()
                self.clear_all_checkpoints()
                if cmd in Command.RESET.value:
                    self._send_message(MessageType.HARD_RESET_PERFORMED)
                else:
                    self._send_message(MessageType.RESET_PERFORMED)
                return

            if cmd.startswith(f"{Command.START.value}:"):
                self._handle_start_command(cmd)
                return

            self._send_error(f"UnknownCmd:{cmd}")

        except Exception as e:
            logger.exception(f"Error processing command {cmd}")
            self._send_error(f"CommandError:{str(e)}")

    def _handle_start_command(self, cmd: str) -> None:
        """Handle START command."""
        try:
            parts = cmd.split(":", 1)
            if len(parts) != 2:
                raise ValueError("Invalid START command format")

            rounds = int(parts[1])
            if rounds <= 0:
                raise ValueError("Number of rounds must be positive")

        except (ValueError, IndexError) as e:
            self._send_error(f"InvalidStartCmd:{str(e)}")
            return

        # Stop any existing simulation
        self._graceful_stop_simulation()

        # Start new simulation
        self._stop_event.clear()
        self._sim_thread = threading.Thread(
            target=self._run_flower_simulation,
            args=(rounds,),
            daemon=True,
            name="VFL-Simulation"
        )
        self._sim_thread.start()

    def _graceful_stop_simulation(self) -> None:
        """Stop the current simulation gracefully."""
        if self._sim_thread and self._sim_thread.is_alive():
            logger.info("Stopping current simulation")
            self._stop_event.set()
            self._sim_thread.join(timeout=10.0)

            if self._sim_thread.is_alive():
                logger.warning("Simulation thread did not terminate within timeout")
            else:
                logger.info("Simulation stopped successfully")

            self._sim_thread = None

    def _run_flower_simulation(self, rounds: int) -> None:
        """Run a single Flower simulation in its own thread."""
        latest_accuracy: Optional[float] = None

        def on_round_complete(rnd: int, metrics: Dict[str, float]) -> None:
            nonlocal latest_accuracy
            latest_accuracy = metrics.get("accuracy")
            embedding_shapes = metrics.get("embedding_shapes", [])
            embedding_sums = metrics.get("embedding_sums", [])
            gradient_shapes = metrics.get("gradient_shapes", [])
            gradient_sums = metrics.get("gradient_sums", [])

            emb_shape_str = ",".join(f"{N}x{D}" for N, D in embedding_shapes)
            grad_shape_str = ",".join(f"{N}x{D}" for N, D in gradient_shapes) # => "128x4,128x4,128x4"

            # print(F"\n\n\nIn round complete === embedding shapes: {emb_shape_str}, gradient shapes: {grad_shape_str}")
            # print(F"In round complete ===embedding sums: {embedding_sums}, gradient sums: {gradient_sums}\n\n\n")
            acc_str = f"{latest_accuracy:.4f}" if latest_accuracy is not None else "na"
            self._send_message(MessageType.ROUND_DONE, rnd, acc_str, emb_shape_str, grad_shape_str)

        rounds += 1  # Randomly extend rounds by -1 to +1 for variability

        if rounds < 1:
            rounds = 1

        try:
            # Prepare data
            labels, _ = process_dataset()
            labels = labels["Survived"].tolist()

            # Create strategy with reporting
            strategy = ReportingStrategy(
                report_fn=on_round_complete,
                stop_evt=self._stop_event,
                labels=labels,
            )

            # Create server app
            server_app = ServerApp(
                server_fn=lambda _: ServerAppComponents(
                    strategy=strategy,
                    config=ServerConfig(num_rounds=rounds),
                )
            )

            # Notify start
            self._send_message(MessageType.TRAINING_STARTED)
            logger.info(f"Starting FL simulation with {rounds} rounds")

            # Run simulation
            run_simulation(
                server_app=server_app,
                client_app=client_app,
                num_supernodes=3,
            )

            logger.info("Simulation completed successfully")

        except StopSimulation:
            logger.info("Simulation stopped by request")
        except Exception as e:
            logger.exception("Error during simulation")
            self._send_error(f"SimulationError:{str(e)}")
        finally:
            # Send completion message
            if latest_accuracy is not None:
                self._send_message(MessageType.TRAINING_DONE, f"{latest_accuracy:.4f}")
            else:
                self._send_message(MessageType.TRAINING_DONE)