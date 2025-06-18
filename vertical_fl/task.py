from pathlib import Path
from logging import WARN
import numpy as np
import pandas as pd
import torch.nn as nn
from flwr.common.logger import log
from datasets import Dataset
from flwr_datasets.partitioner import IidPartitioner

# Number of vertical feature splits
NUM_VERTICAL_SPLITS = 3


def _bin_age(age_series: pd.Series) -> pd.Series:
    bins = [-np.inf, 10, 40, np.inf]
    labels = ["Child", "Adult", "Elderly"]
    return (
        pd.cut(age_series, bins=bins, labels=labels, right=True)
        .astype(str)
        .replace("nan", "Unknown")
    )


def _extract_title(name_series: pd.Series) -> pd.Series:
    titles = name_series.str.extract(r" ([A-Za-z]+)\.", expand=False)
    rare_titles = {
        "Lady", "Countess", "Capt", "Col", "Don", "Dr",
        "Major", "Rev", "Sir", "Jonkheer", "Dona",
    }
    # Group rare titles, normalize others
    titles = titles.replace(list(rare_titles), "Rare")
    titles = titles.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    return titles


def _create_features(df: pd.DataFrame) -> tuple[pd.DataFrame, set]:
    df = df.copy()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Age"] = _bin_age(df["Age"])
    df["Cabin"] = df["Cabin"].str[0].fillna("Unknown")
    df["Title"] = _extract_title(df["Name"])
    df.drop(columns=["PassengerId", "Name", "Ticket"], inplace=True)

    all_features = set(df.columns)
    df = pd.get_dummies(
        df,
        columns=["Sex", "Pclass", "Embarked", "Title", "Cabin", "Age"],
    )
    return df, all_features


def process_dataset(data_path: Path | None = None) -> tuple[pd.DataFrame, set]:
    if data_path is None:
        data_path = Path(__file__).parent.parent / "data" / "train.csv"
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["Embarked", "Fare"])
    return _create_features(df)


def load_data(partition_id: int, num_partitions: int) -> tuple[pd.DataFrame, int]:
    """
    Returns (dataframe, vertical_split_id). Dataframe has no 'Survived' column.
    """
    if num_partitions != NUM_VERTICAL_SPLITS:
        log(
            WARN,
            f"Expected num_partitions={NUM_VERTICAL_SPLITS}, got {num_partitions}",
        )

    df, all_features = process_dataset()
    # Vertical split
    v_partitions = _partition_data_vertically(df, all_features)
    v_split_id = partition_id % NUM_VERTICAL_SPLITS
    v_df = v_partitions[v_split_id]

    # Convert to HuggingFace Dataset for horizontal split
    dataset = Dataset.from_pandas(v_df)
    num_h_partitions = int(np.ceil(num_partitions / NUM_VERTICAL_SPLITS))
    partitioner = IidPartitioner(num_partitions=num_h_partitions)
    partitioner.dataset = dataset

    # Horizontal partition and drop label
    partition = partitioner.load_partition(partition_id % num_h_partitions)
    partition = partition.remove_columns(["Survived"])

    return partition.to_pandas(), v_split_id


def _partition_data_vertically(df: pd.DataFrame, features: set) -> list[pd.DataFrame]:
    # Define three groups of feature keywords
    groups = [
        {"Parch", "Cabin", "Pclass"},
        {"Sex", "Title"},
        features - {"Parch", "Cabin", "Pclass", "Sex", "Title"},
    ]
    partitions = []
    for kws in groups:
        cols = [
            col for col in df.columns
            if any(kw in col for kw in kws) or col == "Survived"
        ]
        partitions.append(df[cols])
    return partitions


class ClientModel(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, 4)

    def forward(self, x):
        return self.fc(x)
