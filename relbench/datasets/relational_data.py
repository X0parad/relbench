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
    url = "mysql+pymysql://guest:relational@db.relational-data.org:3306/"
    task_cls_list = [RelationalDataTask]

    test_timestamp = Timestamp(0)
    val_timestamp = Timestamp(0)
    max_eval_time_frames = None
    

    def __init__(
            self,
            database: str = "financial",
            *,
            process: bool = True,
            save: bool = False,
    ):
        self.database = database
        self.db_url = f"{self.url}{database}?charset=utf8mb4"

        self.name = f"{self.name}-{database}"
        self.db_path = f"./data/{self.name}"
        
        self.process = process
        super().__init__(process=process)

        task = self.get_task('relational-data-task')

        if save:
            self.save_database(task)


    def get_task(self, task_name: str, *args, **kwargs) -> RelationalDataTask:
        if task_name not in self.task_cls_dict:
            raise ValueError(
                f"{self.__class__.name} does not support the task {task_name}."
                f"Please choose from {self.task_names}."
            )
        
        if os.path.exists(f"{self.db_path}/{task_name}.pkl"):
            with open(os.path.join(self.db_path, f'{task_name}.pkl'), 'rb') as f:
               task = pickle.load(f)
        else:
            meta = create_engine(f"{self.url}meta?charset=utf8mb4")
            metadata = read_sql(
                f"SELECT database_name, target_table, target_column, target_id, task FROM meta.information WHERE database_name='{self.database}'",
                meta
                )
            
            # Determine task type based on distinct count
            task_type_dict = {
                'classification': {
                    2: TaskType.BINARY_CLASSIFICATION,
                    'default': TaskType.MULTILABEL_CLASSIFICATION
                },
                'regression': TaskType.REGRESSION
            }

            # Get the base task type from metadata
            base_task_type = metadata['task'][0]
            
            if base_task_type == 'classification':
                # Extract necessary metadata information
                target_table = metadata['target_table'][0]
                target_column = metadata['target_column'][0]
                
                # Query to get the number of distinct values in the target column
                distinct_count_query = f"""
                SELECT COUNT(DISTINCT {target_column}) as distinct_count
                FROM {target_table}
                """

                db = create_engine(self.db_url)

                # Execute the query and fetch the result
                distinct_count_result = read_sql(distinct_count_query, db)
                distinct_count = distinct_count_result['distinct_count'][0]

                task_type = task_type_dict['classification'].get(distinct_count, task_type_dict['classification']['default'])
            else:
                task_type = task_type_dict[base_task_type]


            type_metric_dict = {
                    TaskType.MULTILABEL_CLASSIFICATION : 
                        [multilabel_auprc_micro,
                        multilabel_auprc_macro,
                        multilabel_auroc_micro,
                        multilabel_auroc_macro,],#[roc_auc, accuracy, f1, average_precision],
                    TaskType.REGRESSION : [mae, rmse],
                    TaskType.BINARY_CLASSIFICATION :
                    [roc_auc, accuracy, f1, average_precision],
                }
            
            task = RelationalDataTask(
                    self,
                    entity_col=self.db.table_dict[metadata['target_table'][0]].pkey_col, #metadata['target_id'][0],
                    entity_table=metadata['target_table'][0],
                    target_col=metadata['target_column'][0],
                    task_type=task_type,
                    metrics=type_metric_dict[task_type]
                )
            
            task.mask_test_data(self)
            
        return task
      

    def make_db(self) -> Database:

        if os.path.exists(f"{self.db_path}/db.pkl"):
            with open(os.path.join(self.db_path, 'db.pkl'), 'rb') as f:
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
                
                if len(table.primary_key.columns.values()) > 1:
                    raise Exception("Compound primary key not supported")
                elif len(table.primary_key.columns.values()) == 1:
                    primary_key = tuple(str(pk_column.name) for pk_column in table.primary_key.columns.values())[0]
                else:
                    primary_key=None

                foreign_keys = {}
                for key in inspector.get_foreign_keys(table_name=rel_id):
                    if key["referred_columns"] != key["constrained_columns"]:
                        Exception(f"{key} column names do not match")
                    foreign_keys[key['constrained_columns'][0]] = key['referred_table']

                table_dict[rel_id] = Table(df=df, fkey_col_to_pkey_table=foreign_keys, pkey_col=primary_key, time_col=None)
            
            db = Database(table_dict = table_dict)

            return db


    def save_database(self, task):
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
        # save database
        with open(os.path.join(self.db_path, 'db.pkl'), 'wb') as f:
            pickle.dump(self.db, f)
        # save task
        with open(os.path.join(self.db_path, f'{task.name}.pkl'), 'wb') as f:
            pickle.dump(task, f)



def infer_stypes(database: Database) -> dict[str, dict[str, stype]]:
    inferred_stypes = {}
    for name, table in database.table_dict.items():
        inferred_stypes[name] = infer_df_stype(table.df)
    return inferred_stypes
