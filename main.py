import csv
import re
from typing import List

import pandas as pd
import torch
from simpletransformers.ner import NERModel, NERArgs


def load_data(from_csv: str) -> pd.DataFrame:
    data = {
        'sentence_id': [],
        'words': [],
        'labels': [],
    }

    chars_to_ignore = ',.:'

    with open(from_csv, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)

        for index, row in enumerate(reader):
            sentence_tokenized: List[str] = row[0].split(' ')
            data['sentence_id'] += [index] * len(sentence_tokenized)
            data['words'] += sentence_tokenized

            labels = []
            for token_index, token in enumerate(sentence_tokenized):
                local_labels = []
                for label_index, label in enumerate(header[1:]):
                    if re.sub(f'[{chars_to_ignore}]', '', token) in row[label_index + 1].split(' '):
                        local_labels.append(label)
                if len(local_labels) == 1:
                    labels.append(local_labels[0])
                else:
                    labels.append('O')

            data['labels'] += labels

    return pd.DataFrame(data=data)


if __name__ == '__main__':
    model_type = 'bert'  # 'roberta'
    model_name = 'herokiller/german-bert-finetuned-ner'  # 'roberta-base'

    print('CUDA enabled:', torch.cuda.is_available())

    labels = ['O', 'PER', 'LOC']

    model_args = NERArgs()
    model_args.labels_list = labels
    model_args.num_train_epochs = 30
    model_args.use_multiprocessing = True
    model_args.save_model_every_epoch = False
    model_args.n_gpu = 1
    model_args.best_model_dir = 'models/tf-1'

    model = NERModel(model_type, model_name, use_cuda=True, args=model_args)

    data = load_data('historic_data.csv')
    model.train_model(train_data=data, output_dir='models/', show_running_loss=True)
