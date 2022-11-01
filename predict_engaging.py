import argparse

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import transformers
import wandb
import numpy as np

wandb.init(project="ENGAGING")

df = pd.read_csv('data/GermEval21_TestData.csv')

# make smaller eval file
df = df.sample(frac=0.3, replace=True, random_state=1)

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
                    true_label = df[ 'Sub2_Engaging' ].loc[ df[ 'text' ] == sequence ].values[ 0 ]
                    result = classifier(sequence, pos_label, hypothesis_template=hypothesis)
                    positive_score = result[ 'scores' ][ 0 ]

                res[ pos_label ] = round(positive_score, 2)

            res[ "true_label" ] = true_label

            y_pred = 0
            # # Strategy Nr. 1
            #

            tokens = sequence.split()
            signal_words = ['versteh', 'nachfühl', 'hinenversetz', 'empfind']
            for token in tokens:
              for signal_word in signal_words:
                if signal_word in token:
                  y_pred = 1
                  strategy_1_total += 1



                  if y_pred == true_label:
                    strategy_1_true += 1
                  else:
                    strategy_1_false += 1
                  y_preds.append(y_pred)
                  true_labels.append(true_label)
                  continue

            # Höfliche Anreden
            # Strategy Nr. 2
            if sequence.startswith(('Lieber', 'Liebes', 'Liebe')):
                y_pred = 1
                strategy_2_total += 1

                if y_pred == true_label:
                    strategy_2_true += 1
                else:
                    strategy_2_false += 1
                y_preds.append(y_pred)
                true_labels.append(true_label)
                continue

            # Strategy Nr. 3
            if res[ "Respektbekundung" ] > 0.2:
                if res[ 'Ironie' ] > 0.6:
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
            if res[ 'externe Quelle' ] > 0.1:
                if res[ 'Tatsachenbehauptung' ] > threshold:
                    strategy_4_total += 1
                    y_pred = 0

                    if y_pred == true_label:
                        strategy_4_true += 1
                    else:
                        strategy_4_false += 1
                    y_preds.append(y_pred)
                    true_labels.append(true_label)
                    continue

            if res[ 'Lösungsvorschlag' ] > 0.6:
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

        wandb.log({'accuracy': acc, 'precision': prec, 'recall': recall, 'f1': f1, "Strategies": "1-4"})
        print(f'Accuracy: {acc}\n Precision: {prec}\n Recall: {recall}\n F1-Score: {f1}')
        print(
            f'Total Strategy 1: {strategy_1_total}: True = {strategy_1_true}, False = {strategy_1_false}')

        print(
            f'Total Strategy 2: {strategy_2_total}: True = {strategy_2_true}, False = {strategy_2_false}')
        print(
            f'Total Strategy 3: {strategy_3_total}: True = {strategy_3_true}, False = {strategy_3_false} ')
        print(
            f'Total Strategy 4: {strategy_4_total}: True = {strategy_4_true}, False = {strategy_4_false} ')
        print(
            f'Base Strategy: {base_strategy_total}: True = {base_strategy_true}, False = {base_strategy_false}')

def main(args):
    wandb.run.name = args.name

    config = {'pos': {"Respektbekundung": ["Dieser Kommentar ist eine {}"],
                 "Ironie": ["Dieser Kommentar enthält {}"],
                 'externe Quelle': ["Dieser Kommentar ist eine {}"],
                 'Tatsachenbehauptung': ["Dieser Kommentar ist eine {}"],
                 'Lösungsvorschlag': ["Dieser Kommentar ist eine {}"]
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