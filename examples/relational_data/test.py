
from relbench.datasets import dataset_names, get_dataset
print(dataset_names)
dataset = get_dataset("relational-data", database='financial')
#dataset.pack_db(root='./data')
print(dataset.task_names)