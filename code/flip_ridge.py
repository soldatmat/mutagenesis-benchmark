import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

FILE_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(FILE_PATH / 'FLIP' / 'baselines'))

from utils import *
from evals import regression_eval
from train import *
from models import RidgeRegression

results_path = FILE_PATH / '..' / 'results' / 'flip'

model = "ridge"

alpha = 1.0

def evaluate_ridge(X, y, model, SAVE_PATH, sequences):
    out = model.predict(X)
    #rho, mse = regression_eval(predicted=out, labels=y, SAVE_PATH=SAVE_PATH)

    data_dict = {'sequence': sequences, 'score': y, 'prediction': out}
    df = pd.DataFrame(data_dict)

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAVE_PATH / "predictions.csv")

def run_ridge(train, test):
    EVAL_PATH = results_path / dataset / model
    EVAL_PATH = EVAL_PATH if split == "" else EVAL_PATH / split
    EVAL_PATH.mkdir(parents=True, exist_ok=True)

    train_seq, train_target = get_data(train, max_length, encode_pad=False, one_hots=True)
    test_seq, test_target = get_data(test, max_length, encode_pad=False, one_hots=True)
    # initialize model
    lr_model = RidgeRegression(solver='lsqr', tol=1e-4, max_iter=int(1e6), alpha=alpha)
    # train and pass back trained model
    lr_trained, epochs_trained = train_ridge(train_seq, train_target, lr_model)
    
    # TODO evaluate train + test data
    all_seq = np.concatenate([train_seq, test_seq])
    all_target = np.concatenate([train_target, test_target])
    all_sequences = train.sequence.values.tolist() + test.sequence.values.tolist()
    evaluate_ridge(all_seq, all_target, lr_trained, EVAL_PATH, all_sequences)

if __name__ == '__main__':
    datasets = ["GB1", "PhoQ", "TrpB"]
    splits = ["non-zero_medium_gap0_short", "non-zero_hard_gap0_short"]

    for dataset in datasets:
        for split in splits:
            data_path = FILE_PATH / '..' / 'data' / 'combinatorial' / dataset / ('flip_' + split + '.csv')
            df = pd.read_csv(data_path)
            val_split = False
            train, test, max_length = prepare_dataset(df, val_split)
            run_ridge(train, test)
