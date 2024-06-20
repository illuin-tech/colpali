from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

import pandas as pd


class EvalManager:
    """
    Stores evaluation results for various datasets and metrics.

    The data is stored in a pandas DataFrame with a MultiIndex for columns.
    The first level of the MultiIndex is the dataset name and the second level is the metric name.

    Usage:
    >>> evaluator = Evaluator.from_dirpath("data/evaluation_results/")
    >>> print(evaluator.data)

    """

    model_col: ClassVar[str] = "model"
    dataset_col: ClassVar[str] = "dataset"
    metric_col: ClassVar[str] = "metric"

    def __init__(self, data: Optional[pd.DataFrame] = None):
        if data is None:
            data = pd.DataFrame()
        self._df = data
        self._df.index = self._df.index.rename(EvalManager.model_col)

    def __str__(self) -> str:
        return self.data.__str__()

    @staticmethod
    def from_dict(data: Dict[Any, Any]) -> EvalManager:
        """
        Load evaluation results from a dictionary.

        Expected format:
        {
            "model1": pd.read_json(path1).T.stack(),
            "model2": pd.read_json(path2).T.stack(),
        }

        """
        df = pd.DataFrame.from_dict(data, orient="index")
        return EvalManager(df)

    @staticmethod
    def from_json(path: str | Path) -> EvalManager:
        datapath = Path(path)
        if not datapath.is_file():
            raise FileNotFoundError(f"{path} is not a file")
        data = {}
        data[datapath.stem] = pd.read_json(datapath).T.stack()  # pylint: disable=no-member
        return EvalManager.from_dict(data)

    @staticmethod
    def from_dir(datadir: str | Path) -> EvalManager:
        datadir_ = Path(datadir)
        if not datadir_.is_dir():
            raise FileNotFoundError(f"{datadir} is not a directory")

        eval_files = list(datadir_.glob("*.json"))

        data = {}

        for filepath in eval_files:
            data[filepath.stem] = pd.read_json(filepath).T.stack()  # pylint: disable=no-member

        return EvalManager.from_dict(data)

    @staticmethod
    def from_csv(path: str | Path) -> EvalManager:
        """
        Load evaluation results from a CSV file.
        """
        try:
            df = pd.read_csv(path, index_col=0, header=[0, 1])
            return EvalManager(df)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            raise e

    @property
    def data(self) -> pd.DataFrame:
        """
        Returns the evaluation results as a pandas DataFrame.
        """
        return self._df.copy()

    @property
    def models(self) -> pd.Index:
        """
        Returns the models for which there are evaluation results.
        """
        return self.data.index

    @property
    def datasets(self) -> pd.Index:
        """
        Returns the datasets for which there are evaluation results.
        """
        return self.data.columns.get_level_values(0).unique()

    @property
    def metrics(self) -> pd.Index:
        """
        Returns the metrics for which there are evaluation results.
        """
        return self.data.columns.get_level_values(1)

    @staticmethod
    def melt(df: pd.DataFrame) -> pd.DataFrame:
        """
        Melt a suitable DataFrame (e.g. returned by `get_df_for_dataset` and
        `get_df_for_metric`) into a 'long' format.
        """
        return df.T.reset_index(names=[EvalManager.dataset_col, EvalManager.metric_col]).melt(
            id_vars=[EvalManager.dataset_col, EvalManager.metric_col],
            var_name=EvalManager.model_col,
            value_name="score",
        )

    @property
    def melted(self) -> pd.DataFrame:
        """
        Returns the evaluation results as a 'melted' DataFrame.
        Useful for plotting with seaborn.
        """
        return EvalManager.melt(self.data)

    def get_df_for_model(self, model: str) -> pd.DataFrame:
        if model not in self.data.index:
            raise ValueError(f"Model {model} not found in the evaluation results")
        return self.data.loc[[model], :]  # type: ignore

    def get_df_for_dataset(self, dataset: str) -> pd.DataFrame:
        if dataset not in self.datasets:
            raise ValueError(f"Dataset {dataset} not found in the evaluation results")
        return self.data.loc[:, (dataset, slice(None))]  # type: ignore

    def get_df_for_metric(self, metric: str) -> pd.DataFrame:
        if metric not in self.metrics:
            raise ValueError(f"Metric {metric} not found in the evaluation results")
        return self.data.loc[:, (slice(None), metric)]  # type: ignore

    def sort_by_dataset(self, ascending: bool = True) -> EvalManager:
        """
        Sort the evaluation results by dataset name.
        """
        df = self.data.T.sort_index(level=0, ascending=ascending).T
        return EvalManager(df)

    def sort_by_metric(self, ascending: bool = True) -> EvalManager:
        """
        Sort the evaluation results by metric name.
        """
        df = self.data.T.sort_index(level=1, ascending=ascending).T
        return EvalManager(df)

    def sort_columns(self, ascending: bool = True) -> EvalManager:
        """
        Sort the evaluation results by dataset name and then by metric name.
        """
        df = self.data.T.sort_index(level=[0, 1], ascending=ascending).T
        return EvalManager(df)

    def to_csv(self, path: str | Path):
        """
        Save the evaluation results to a CSV file.

        Using `Evaluation.from_csv(path_to_saved_csv)` will load the evaluation results back into memory.
        """
        savepath = Path(path)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(savepath)
