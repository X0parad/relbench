from relbench.data import Database, RelBenchDataset, Table
from sqlalchemy import create_engine, MetaData, inspect
from pandas import read_sql, Timestamp
from relbench.data.task_node import TaskType
from relbench.tasks.relational_data_task import InDatabaseColumnTask
from torch_frame.utils import infer_df_stype
from torch_frame import stype

from relbench.metrics import accuracy, average_precision, f1, mae, rmse, roc_auc, multilabel_auprc_macro, multilabel_auprc_micro, multilabel_auroc_macro, multilabel_auroc_micro

import os
import pickle

#requires pymysql

class RelationalDataTask(InDatabaseColumnTask):
    name = 'relational-data-task'



class RelationalData(RelBenchDataset):
    name = "relational-data"
    url = "mysql+pymysql://guest:relational@db.relational-data.org/"
    task_cls_list = [RelationalDataTask]

    test_timestamp = Timestamp(0)
    val_timestamp = Timestamp(0)
    max_eval_time_frames = None

    def __init__(
            self,
            database: str = "financial",
            *,
            process: bool = True,
    ):
        self.database = database
        self.db_url = f"{self.url}{database}?charset=utf8mb4"

        self.name = f"{self.name}-{database}"
        
        super().__init__(process=process)

        self.get_task('relational-data-task')


    def get_task(self, task_name: str, *args, **kwargs) -> RelationalDataTask:
        if task_name not in self.task_cls_dict:
            raise ValueError(
                f"{self.__class__.name} does not support the task {task_name}."
                f"Please choose from {self.task_names}."
            )
        
        meta = create_engine(f"{self.url}meta?charset=utf8mb4")
        metadata = read_sql(
            f"SELECT database_name, target_table, target_column, target_id, task FROM meta.information WHERE database_name='{self.database}'",
            meta
            )
        task_type_dict = {
                'classification' : TaskType.MULTILABEL_CLASSIFICATION,
                'regression' : TaskType.REGRESSION
            }
        task_type= task_type_dict[metadata['task'][0]]
        type_metric_dict = {
                TaskType.MULTILABEL_CLASSIFICATION : 
                    [multilabel_auprc_micro,
                    multilabel_auprc_macro,
                    multilabel_auroc_micro,
                    multilabel_auroc_macro,],#[roc_auc, accuracy, f1, average_precision],
                TaskType.REGRESSION : [mae, rmse]
            }
        
        task = RelationalDataTask(
                self,
                entity_col=metadata['target_id'][0],
                entity_table=metadata['target_table'][0],
                target_col=metadata['target_column'][0],
                task_type=task_type,
                metrics=type_metric_dict[task_type]
            )
        
        task.mask_test_data(self)

        return task
      

    def make_db(self) -> Database:
        db_path = f"./data/{self.name}"

        if os.path.exists(f"{db_path}/db.pkl"):
            with open(os.path.join(db_path, 'db.pkl'), 'rb') as f:
               return pickle.load(f)
        else:
            "retrieve data from server"

            db = create_engine(self.db_url)

            metadata = MetaData()
            metadata.reflect(bind=db)
            inspector = inspect(db)

            table_dict={}
            for rel_id in metadata.tables.keys():
                df = read_sql(f"SELECT * FROM `{rel_id}`", db)

                table = metadata.tables[rel_id]

                primary_key = tuple(str(pk_column.name) for pk_column in table.primary_key.columns.values())[0]

                foreign_keys = {}
                for key in inspector.get_foreign_keys(table_name=rel_id):
                    if key["referred_columns"] != key["constrained_columns"]:
                        print(f"{key} column names do not match")
                    foreign_keys[key['constrained_columns'][0]] = key['referred_table']

                table_dict[rel_id] = Table(df=df, fkey_col_to_pkey_table=foreign_keys, pkey_col=primary_key, time_col=None)
            
            db = Database(table_dict = table_dict)

            # save database
            with open(os.path.join(db_path, 'db.pkl'), 'wb') as f:
                pickle.dump(db, f)

            return db

def infer_stypes(database: Database) -> dict[str, dict[str, stype]]:
    inferred_stypes = {}
    for name, table in database.table_dict.items():
        inferred_stypes[name] = infer_df_stype(table.df)
    return inferred_stypes
