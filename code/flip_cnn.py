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
#from evals import *
from train import *
from models import FluorescenceModel

results_path = FILE_PATH / '..' / 'results' / 'flip'

model = "cnn"

n_epochs = 100
batch_size = 256
kernel_size = 5
input_size = 1024
dropout = 0.0

def evaluate_cnn(data_iterator, model, device, MODEL_PATH, SAVE_PATH):
    """ run data through model and print eval stats """

    model = model.to(device)
    bestmodel_save = MODEL_PATH / 'bestmodel.tar' 
    sd = torch.load(bestmodel_save, weights_only=False)
    model.load_state_dict(sd['model_state_dict'])
    print('loaded the saved model')

    def test_step(model, batch):
        src, tgt, mask = batch
        src = src.to(device).float()
        tgt = tgt.to(device).float()
        mask = mask.to(device).float()
        output = model(src, mask)
        return output.detach().cpu(), tgt.detach().cpu()
    
    model = model.eval()

    outputs = []
    tgts = []
    one_hot = []
    masks = []
    n_seen = 0
    for i, batch in enumerate(data_iterator):
        output, tgt = test_step(model, batch)
        outputs.append(output)
        tgts.append(tgt)
        one_hot.append(batch[0])
        masks.append(batch[2])
        n_seen += len(batch[0])

    out = np.squeeze(torch.cat(outputs).numpy(), axis=1)
    labels = np.squeeze(torch.cat(tgts).cpu().numpy(), axis=1)
    masks = torch.cat(masks).cpu().to(bool)

    sequences = torch.argmax(torch.cat(one_hot).cpu(), dim=2)
    sequences = [sequences[s][masks[s]] for s in range(len(sequences))]
    sequences = [''.join(map(lambda i: vocab[i], seq)) for seq in sequences]

    data_dict = {'sequence': sequences, 'score': labels, 'prediction': out}
    df = pd.DataFrame(data_dict)

    SAVE_PATH.mkdir(parents=True, exist_ok=True) # make directory if it doesn't exist already
    df.to_csv(SAVE_PATH / "predictions.csv")


def run_cnn(train, val, test):
    device = torch.device('cuda:0')

    EVAL_PATH = results_path / dataset / model
    EVAL_PATH = EVAL_PATH if split == "" else EVAL_PATH / split
    EVAL_PATH.mkdir(parents=True, exist_ok=True)

    collate = ASCollater(vocab, Tokenizer(vocab), pad=True)
    train_iterator = DataLoader(SequenceDataset(train), collate_fn=collate, batch_size=batch_size, shuffle=True, num_workers=4)
    val_iterator = DataLoader(SequenceDataset(val), collate_fn=collate, batch_size=batch_size, shuffle=True, num_workers=4)
    #test_iterator = DataLoader(SequenceDataset(test), collate_fn=collate, batch_size=batch_size, shuffle=True, num_workers=4)

    # initialize model
    cnn_model = FluorescenceModel(len(vocab), kernel_size, input_size, dropout)

    # create optimizer and loss function
    optimizer = torch.optim.Adam([
        {'params': cnn_model.encoder.parameters(), 'lr': 1e-3, 'weight_decay': 0},
        {'params': cnn_model.embedding.parameters(), 'lr': 5e-5, 'weight_decay': 0.05},
        {'params': cnn_model.decoder.parameters(), 'lr': 5e-6, 'weight_decay': 0.05}
    ])
    criterion = torch.nn.MSELoss()

    # train and pass back epochs trained - for CNN, save model 
    epochs_trained = train_cnn(train_iterator, val_iterator, cnn_model, device, criterion, optimizer, n_epochs, EVAL_PATH)

    # evaluate
    all_iterator = DataLoader(SequenceDataset(pd.concat([train, val, test])), collate_fn=collate, batch_size=batch_size, shuffle=False, num_workers=4)
    evaluate_cnn(all_iterator, cnn_model, device, EVAL_PATH, EVAL_PATH / 'evaluate')


if __name__ == '__main__':
    # Load an original FLIP data split
    #train, val, test, _ = load_dataset(dataset, split+'.csv')

    datasets = ["GB1", "PhoQ", "TrpB"]
    splits = ["medium_gap0", "hard_gap0"]

    for dataset in datasets:
        for split in splits:
            data_path = FILE_PATH / '..' / 'data' / 'combinatorial' / dataset / ('flip_' + split + '.csv')
            df = pd.read_csv(data_path)
            val_split = True
            train, val, test, max_length = prepare_dataset(df, val_split)

            run_cnn(train, val, test)
