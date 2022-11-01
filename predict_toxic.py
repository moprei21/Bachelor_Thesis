import argparse

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import transformers
import wandb
import numpy as np

wandb.init(project="TOXIC")

df = pd.read_csv('data/GermEval21_TestData.csv')

# make smaller eval file
df = df.sample(frac=0.1, replace=True, random_state=1)

nli_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path='./pretrain_out/.')
tokenizer = AutoTokenizer.from_pretrained("pretrain_out")

classifier = transformers.ZeroShotClassificationPipeline(model=nli_model, tokenizer=tokenizer)

def multi_hypo(config):
    thresholds = np.arange(0.2, 1, 0.2)
    thresholds = [0.2]
    for threshold in thresholds:
        y_preds = [ ]
        true_labels = [ ]
        false_positives = [ ]

        strategy_1_true = 0
        strategy_1_false = 0
        strategy_2_true = 0
        strategy_2_false = 0
        strategy_3_true = 0
        strategy_3_false = 0
        strategy_4_true = 0
        strategy_4_false = 0
        strategy_1_total = 0
        strategy_2_total = 0
        strategy_3_total = 0
        strategy_4_total = 0
        base_strategy_total = 0
        base_strategy_true = 0
        base_strategy_false = 0
        print(f'Threshold: {threshold}')
        for sequence in tqdm(df[ 'text' ].values):
            res = {}
            for pos_label in config[ 'pos' ].keys():
                for hypothesis in config[ 'pos' ][ pos_label ]:
                    true_label = df[ 'label' ].loc[ df[ 'text' ] == sequence ].values[ 0 ]
                    result = classifier(sequence, pos_label, hypothesis_template=hypothesis)
                    positive_score = result[ 'scores' ][ 0 ]

                res[ pos_label ] = round(positive_score, 2)

            res[ "true_label" ] = true_label

            y_pred = 0

            # # Strategy Nr. 1
            # filter for SCREAMING sentences or !!! or ???
            tokens = sequence.split()
            count = 0
            special_tokens = [ '@USER', '@MEDIUM', '@MODERATOR' ]
            if '!!!' in sequence or '???' in sequence:
                y_pred = 1
                strategy_1_total += 1
                y_preds.append(y_pred)
                true_labels.append(true_label)

                if y_pred == true_label:
                    strategy_1_true += 1
                else:
                    strategy_1_false += 1
                continue

            for token in tokens:

                token = token.rstrip(',.?!:"')
                token = token.lstrip('"')

                if token.isupper() and token.isalnum() and token not in special_tokens and len(token) > 3:
                    count += 1
            if count > 1:
                y_pred = 1
                strategy_1_total += 1
                y_preds.append(y_pred)
                true_labels.append(true_label)

                if y_pred == true_label:
                    strategy_1_true += 1
                else:
                    strategy_1_false += 1

                continue

            # Strategy Nr. 2
            if res[ "Zynismus" ] > 0.7:
                if res[ 'humoristisch veranlagt' ] > 0.3:
                    strategy_2_total += 1
                    y_pred = 1

                    y_preds.append(y_pred)
                    true_labels.append(true_label)

                    if y_pred == true_label:
                        strategy_2_true += 1
                    else:
                        strategy_2_false += 1
                    continue

            # Strategy Nr. 3
            if res[ 'Beleidigung' ] > 0.5:
                if res[ 'humoristisch veranlagt' ] > 0.3:
                    strategy_3_total += 1
                    y_pred = 0

                    if y_pred == true_label:
                        strategy_3_true += 1
                    else:
                        strategy_3_false += 1
                    y_preds.append(y_pred)
                    true_labels.append(true_label)
                    continue

            # Strategy Nr. 4

            if res[ 'Diskriminierung' ] > 0.8:
                if res[ 'gegen' ] > 0.8:
                    strategy_4_total += 1
                    y_pred = 1

                    if y_pred == true_label:
                        strategy_4_true += 1
                    else:
                        strategy_4_false += 1
                    y_preds.append(y_pred)
                    true_labels.append(true_label)
                    continue

            # BASE Strategy

            if res[ 'Beleidigung' ] > 0.5:
                y_pred = 1
            else:
                y_pred = 0
            base_strategy_total += 1

            if y_pred == true_label:
                base_strategy_true += 1
            else:
                base_strategy_false += 1

            y_preds.append(y_pred)
            true_labels.append(true_label)

        f1 = f1_score(true_labels, y_preds, average='macro')
        acc = accuracy_score(true_labels, y_preds)

        prec = precision_score(true_labels, y_preds)
        recall = recall_score(true_labels, y_preds)
        print(f'Accuracy: {acc}\n Precision: {prec}\n Recall: {recall}\n F1-Score: {f1}')


        print(
            f'Total Strategy 1: {strategy_1_total}: True = {strategy_1_true}, False = {strategy_1_false} ')
        print(
            f'Total Strategy 2: {strategy_2_total}: True = {strategy_2_true}, False = {strategy_2_false}')
        print(
            f'Total Strategy 3: {strategy_3_total}: True = {strategy_3_true}, False = {strategy_3_false}')
        print(
            f'Total Strategy 4: {strategy_4_total}: True = {strategy_4_true}, False = {strategy_4_false}')
        print(
            f'Base Strategy: {base_strategy_total}: True = {base_strategy_true}, False = {base_strategy_false}')

        wandb.log({'accuracy': acc, 'precision': prec, 'recall': recall, 'f1': f1, "Strategies": "1-4" })

        print(f'Accuracy: {acc}\n Precision: {prec}\n Recall: {recall}\n F1-Score: {f1}')


def main(args):
    wandb.run.name = args.name

    config = {'pos':
                        {
                         "Beleidigung": ["Dieser Kommentar ist eine {}"],
                         "humoristisch veranlagt": ["Dieser Kommentar ist {}"],
                         "Zynismus": ["Dieser Kommentar ist {}"],
                         "Diskriminierung": ["In diesem Kommentar findet man {}"],
                         "gegen": ["Dieser Kommentar ist {} eine Gruppe gerichtet"]
                                    }
                        }


    wandb.run.config[ 'hypos' ] = config[ 'pos' ]

    multi_hypo(config)

    return config



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help="name of wandb run")

    args = parser.parse_args()

    main(args)