import pandas as pd
from pandas import merge
import py_entitymatching as em
import random

def load_data(left_file_name, right_file_name, label_file_name, blocking_fn, include_self_join=False, seed=42):
    A = em.read_csv_metadata(left_file_name , key="id", encoding='iso-8859-1')
    B = em.read_csv_metadata(right_file_name , key="id", encoding='iso-8859-1')
    columns = A.columns.tolist()
    random.seed(seed)
    random.shuffle(columns)
    A = A[columns]
    B = B[columns]
    em.set_key(A, 'id')
    em.set_key(B, 'id')

    try:
        G = pd.read_csv(label_file_name)
    except:
        G=None
    C = blocking_fn(A, B)

    if include_self_join:
        C_A = blocking_fn(A, A)
        C_B = blocking_fn(B, B)
        return A, B, G, C, C_A,C_B
    else:
        return A, B, G, C
