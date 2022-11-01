import csv
import random
import argparse


hypo_toxic = ["Dieser Kommentar ist eine Beleidigung", "Dieser Kommentar ist humoristisch veranlagt", "Dieser Kommentar ist Zynismus", "In diesem Kommentar findet man Diskriminierung", "Dieser Kommentar ist gegen eine Gruppe gerichtet"]
hypo_engaging = ["Dieser Kommentar ist eine Respektbekundung", "Dieser Kommentar enthält Ironie", "Dieser Kommentar ist eine externe Quelle", "Dieser Kommentar ist eine Tatsachenbehauptung", "Dieser Kommentar ist ein Lösungsvorschlag"]
hypo_fact =[ "Dieser Kommentar ist eine externe Quelle", "Dieser Kommentar ist eine Tatsachenbehauptung","Dieser Kommentar ist ein Wahrheitsanspruch"]



def get_random_entries(num,task):
    with open('data/GermEval21_TrainData.csv', 'r', encoding='utf-8') as train_data:
        task_table = {"t": 2, "e": 3, "f": 4}
        csvreader = csv.reader(train_data)
        entries_pos = []
        entries_neg = []
        for row in csvreader:
            if row[task_table[task]] == '0':

                entries_neg.append(row)
            else:
                entries_pos.append(row)
        num_neg =int(0.625 * num)
        num_pos = int(num - num_neg)
        subset_entries_pos = random.sample(entries_pos,num_pos)
        subset_entries_neg = random.sample(entries_neg,num_neg)
        subset_entries = subset_entries_neg+subset_entries_pos
        print(subset_entries)
        return subset_entries


def create_hypo_dataset(hypothesis, subset, outfile_name, task):
    task_table = {"t":2, "e":3, "f":4}
    with open(outfile_name, 'w', encoding='utf-8',newline='') as outfile:
        writer = csv.writer(outfile)
        header = [ 'premise', 'hypothesis', 'label' ]
        writer.writerow(header)
        for row in subset:
            if row[task_table[task]] == '0':
                label = '2'
            else:
                label = '0'

            hypo = random.choice(hypothesis)
            entry = [row[1],hypo,label]
            writer.writerow(entry)

def main(args):
    subset = get_random_entries(args.num, args.task)
    if args.task == "t":
        hypo = hypo_toxic
    elif  args.task == "e":
        hypo = hypo_engaging
    else:
        hypo = hypo_fact
    create_hypo_dataset(hypo, subset, args.outfile,args.task)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, help='number of comments in subset of datafile')
    parser.add_argument('--task', type=str,choices=["t","e","f"] , help="name of subtask to choose")
    parser.add_argument('--outfile', type=str, help='path of outfile')

    args = parser.parse_args()

    main(args)


