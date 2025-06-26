from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from vertical_fl.strategy import Strategy
from vertical_fl.task import process_dataset



def server_fn(context: Context) -> ServerAppComponents:
    """
    Construct components that configure the Flower ServerApp behaviour.

    Reads server configuration from context.run_config:
    - "num-server-rounds": int, number of federated rounds to run (default: 1).
    - "learning-rate": float, server-side learning rate for aggregation (default: 0.01).

    Returns:
        ServerAppComponents with custom strategy and server config.
    """
    # Load processed dataset and extract labels
    df, _ = process_dataset()
    labels = df["Survived"].values.tolist()

    # Fetch server run parameters, with sensible defaults
    num_rounds = int(context.run_config.get("num-server-rounds", 1))
    lr = float(context.run_config.get("learning-rate", 0.01))

    # Initialize custom vertical-FL strategy
    strategy = Strategy(labels=labels, lr=lr)

    # Configure number of rounds for the server
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Launch the Flower server
app = ServerApp(server_fn=server_fn)
