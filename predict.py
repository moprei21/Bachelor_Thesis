import argparse

import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import transformers
import wandb
import numpy as np

wandb.init(project="my-test-project")

df = pd.read_csv('data/GermEval21_TestData.csv')

# make smaller eval file
df = df.sample(frac=0.1, replace=True, random_state=1)

nli_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path='./pretrain_out/.')
tokenizer = AutoTokenizer.from_pretrained("pretrain_out")

classifier = transformers.ZeroShotClassificationPipeline(model=nli_model, tokenizer=tokenizer)

config_toxic = {'pos_label': [ 'Beleidigung' ],
                'neg_label': '',
                'hypo': 'Dieser Text enthält eine {}',
                'task': 'label',
                'threshold': 0.5,
                'multi_class': False}

wandb.config = config_toxic


def multi_hypo(config):
    res = [ ]
    true_labels = [ ]
    for sequence in tqdm(df[ 'text' ].values):
        num_labels = len(config[ 'pos_label' ])
        true_label = df[ config[ 'task' ] ].loc[ df[ 'text' ] == sequence ].values[ 0 ]
        # print(f'\n\nNEW sequence: {sequence}')
        pos_sequence_probability = 0
        neg_sequence_probability = 0
        for positive_label in config[ 'pos_label' ]:
            result = classifier(sequence, positive_label, hypothesis_template=config[ 'hypo' ],
                                multi_label=config[ 'multi_class' ])
            labels = result[ 'labels' ]
            probs = result[ 'scores' ]
            pos_probability = probs[ 0 ]
            # print(labels, probs)

            pos_sequence_probability += pos_probability

            # if probability >= sequence_probability:
            # sequence_probability = probability

        pos_sequence_probability = pos_sequence_probability / num_labels

        y_pred = 0
        if pos_sequence_probability > config[ 'threshold' ]:
            y_pred = 1

        # print(f'True label: {true_label}, Y_Pred: {y_pred}\n')
        res.append(y_pred)
        true_labels.append(true_label)

    return res


def eval_zero(predicted_labels, config):
    true_labels = [ ]
    for true_label in df[ config[ 'task' ] ].values:
        true_labels.append(true_label)

    f1 = f1_score(true_labels, predicted_labels, average='macro')
    acc = accuracy_score(true_labels, predicted_labels)

    prec = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)

    wandb.log({'accuracy': acc, 'precision': prec, 'recall': recall, 'f1': f1})

    print(f'Accuracy: {acc}\n Precision: {prec}\n Recall: {recall}\n F1-Score: {f1}')


def main(args):
    wandb.run.name = args.name
    thresholds = np.arange(0, 1, 0.1)
    for threshold in thresholds:
        config_toxic = {'pos_label': [ 'Beleidigung' ],
                        'neg_label': '',
                        'hypo': 'Dieser Text enthält eine {}',
                        'task': 'label',
                        'threshold': threshold,
                        'multi_class': False}

        config_engaging = {'pos_label': [ 'persönliche Erfahrung' ],
                           'neg_label': '',
                           'hypo': 'Dieser Text ist eine {}',
                           'task': 'Sub2_Engaging',
                           'threshold': threshold,
                           'multi_class': False}

        config_fact = {'pos_label': [ 'externe Quelle' ],
                       'neg_label': '',
                       'hypo': 'Dieser Text ist eine {}',
                       'task': 'Sub3_FactClaiming',
                       'threshold': threshold,
                       'multi_class': False}

        if args.config == 'toxic':
            config = config_toxic
        elif args.config == 'engaging':
            config = config_engaging
        else:
            config = config_fact

        wandb.run.config[ 'hypo' ] = config[ 'hypo' ]
        wandb.run.config[ 'pos_label' ] = config[ 'pos_label' ]

        predicted_labels = multi_hypo(config)
        eval_zero(predicted_labels, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, help="name of wandb run")
    parser.add_argument('--config', type=str, help='name of config')

    args = parser.parse_args()

    main(args)
