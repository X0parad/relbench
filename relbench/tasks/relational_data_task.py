from typing import List, Callable, Any, Dict, Optional, Literal

import pandas as pd
import numpy as np
from numpy._typing import NDArray
import torch

import torch_frame
from torch_frame import stype
from torch_frame.data.dataset import DataFrameToTensorFrameConverter
from torch_frame.data.stats import compute_col_stats, StatType
from torch_frame.utils.split import generate_random_split, SPLIT_TO_NUM

from relbench.data.dataset import Dataset, Database
from relbench.data.table import Table

from relbench.data.task_node import NodeTask, TaskType



RelBenchTaskToStype = {
    TaskType.REGRESSION: stype.numerical,
    TaskType.BINARY_CLASSIFICATION: stype.categorical,
}


class InDatabaseColumnTask(NodeTask):

    name: str

    task_dir: str = "tasks"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 1729

    def __init__(
                self,
                dataset: Dataset,
                entity_col: str,
                entity_table: str,
                target_col: str,
                task_type: TaskType,
                metrics: List[Callable[[NDArray, NDArray], float]],
                **kwargs
            ) -> None:
        
        super(InDatabaseColumnTask, self).__init__(
            dataset=dataset,
            timedelta=pd.Timedelta(0),
            target_col=target_col,
            entity_table=entity_table,
            entity_col=entity_col,
            metrics=metrics,
        )

        self.task_type = task_type
        self.entity_table = entity_table
        self.stype = RelBenchTaskToStype[task_type]
        self.col_stats = compute_col_stats(dataset.db.table_dict[entity_table].df[target_col], stype=self.stype)
        
        self.mapper = get_frame_mapper(
            col_name=target_col,
            col_stats=self.col_stats,
            col_stype=self.stype,
        )

    #@classmethod
    def mask_test_data(self, dataset: Dataset) -> None:
        table = dataset.db.table_dict[self.entity_table]
        if table is dataset._full_db.table_dict[self.entity_table]:
            table = Table(
                df=table.df.copy(deep=True),
                fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
                pkey_col=table.pkey_col,
                time_col=table.time_col,
            )
            dataset.db.table_dict[self.entity_table] = table

        num_rows = table.df.shape[0]
        split = generate_random_split(
            length=num_rows,
            seed=self.seed,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
        )
        test_mask = split == SPLIT_TO_NUM['test']

        table.df = table.df.copy(deep=True)
        table.df.loc[test_mask, self.target_col] = None
        table.df['<SPLIT>'] = split

    def make_table(
        self,
        db: Database,
        split: Literal['train', 'val', 'test'],
        split_db: Optional[Database] = None,
    ) -> Table:
        r"""To be implemented by subclass."""
        table = db.table_dict[self.entity_table]
        input_cols = [self.entity_col, self.target_col]

        df = table.df
        if split_db is None:
            split_col = df['<SPLIT>']
        else:
            split_col = split_db.table_dict[self.entity_table].df['<SPLIT>']

        mask = split_col == SPLIT_TO_NUM[split]
        df = df[input_cols][mask].copy(deep=True)

        if self.stype == stype.categorical:
            x = self.mapper.forward(df[self.target_col]) 
            df[self.target_col] = x.numpy()
        
        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=table.pkey_col,
            time_col=table.time_col,
        )

    @property
    def train_table(self) -> Table:
        """Returns the train table for a task."""
        if 'train' not in self._cached_table_dict:
            table = self.make_table(self.dataset.db, 'train')
            self._cached_table_dict['train'] = table
        else:
            table = self._cached_table_dict['train']
        return table

    @property
    def val_table(self) -> Table:
        r"""Returns the val table for a task."""
        if 'val' not in self._cached_table_dict:
            table = self.make_table(self.dataset.db, 'val')
            self._cached_table_dict['val'] = table
        else:
            table = self._cached_table_dict['val']
        return table

    def _mask_input_cols(self, table: Table) -> Table:
        input_cols = [
            *table.fkey_col_to_pkey_table.keys(),
        ]
        return Table(
            df=table.df[input_cols],
            fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
            pkey_col=table.pkey_col,
            time_col=table.time_col,
        )

    @property
    def test_table(self) -> Table:
        r"""Returns the test table for a task."""
        if 'full_test' not in self._cached_table_dict:
            table = self.make_table(
                self.dataset._full_db.upto(self.dataset.test_timestamp),
                'test',
                self.dataset.db
            )
            self._cached_table_dict['full_test'] = table
        else:
            table = self._cached_table_dict['full_test']
        self._full_test_table = table
        return self._mask_input_cols(self._full_test_table)
    
def get_frame_mapper(col_name: str, col_stats: Dict[StatType, Any], col_stype: stype, sep: Optional[str] = ','):
    converter = DataFrameToTensorFrameConverter(
        col_to_stype={col_name: col_stype},
        col_stats={col_name: col_stats},
        col_to_sep={col_name: sep}
    )
    mapper = converter._get_mapper(col_name)
    return mapper