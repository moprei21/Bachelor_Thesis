import csv
import random
import argparse

def get_random_entries(num):
    with open('data/GermEval21_TrainData.csv', 'r', encoding='utf-8') as train_data:
        csvreader = csv.reader(train_data)
        entries = []
        for row in csvreader:
            entries.append(row)

        subset_entries = random.sample(entries,num)
        return subset_entries


def create_hypo_dataset(hypothesis, subset, outfile_name):
    with open(outfile_name, 'w', encoding='utf-8',newline='') as outfile:
        writer = csv.writer(outfile)
        header = [ 'premise', 'hypothesis', 'label' ]
        writer.writerow(header)
        for row in subset:
            if row[2] == '0':
                label = '2'
            else:
                label = '0'
            entry = [row[1],hypothesis,label]
            writer.writerow(entry)

def main(args):
    subset = get_random_entries(args.num)
    create_hypo_dataset(args.h, subset, args.outfile)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, help='number of comments in subset of datafile')
    parser.add_argument('--h', type=str, help='hypothesis that nli dataset should be created on')
    parser.add_argument('--outfile', type=str, help='path of outfile')

    args = parser.parse_args()

    main(args)


