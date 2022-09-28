# finetuning the models on different input sizes
import transformers
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import evaluate

model = AutoModelForSequenceClassification.from_pretrained("deepset/gbert-large")
tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-large")
dataset = load_dataset("csv", data_files={"train": ["data/GermEval21_TrainData.csv"], "test": "data/GermEval21_TestData.csv"})


metric = evaluate.load('accuracy')

def tokenize_function(examples):
    return tokenizer(examples[ "comment_text" ], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(32))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(32))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



def finetuning_trainer():
    training_args = TrainingArguments(num_train_epochs=5,
                                      evaluation_strategy='epoch',
                                      save_strategy='epoch',
                                      output_dir= 'test_trainer',
                                      learning_rate=0.0001,
                                )
    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      train_dataset=small_train_dataset,
                      eval_dataset=small_eval_dataset,
                      compute_metrics=compute_metrics,
                    )
    trainer.train()
    trainer.save_pretrained('model')


device = 'cpu'

finetuning_trainer()

def zero_shot_setting():
    classifier = transformers.ZeroShotClassificationPipeline(model=AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path='./model/.'), tokenizer=AutoTokenizer.from_pretrained("deepset/gbert-large"))
    sequence = "Letzte Woche gab es einen Selbstmord in einer nahe gelegenen kolonie"
    candidate_labels = [ "Verbrechen", "Tragödie", "Stehlen" ]
    hypothesis_template = "In deisem geht es um {}."  ## Since monolingual model,its sensitive to hypothesis template. This can be experimented

    print(classifier(sequence, candidate_labels, hypothesis_template=hypothesis_template))


def entailment():
    nli_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path='./model/.')
    tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-large")

    premise = 'Ich liebe alle Personen auf der Welt'
    hypothesis = 'Dieser Text enthält Hass'

    x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                     truncation_strategy='only_first')
    logits = nli_model(x.to(device))[0]
# we throw away "neutral" (dim 1) and take the probability of
# "entailment" (2) as the probability of the label being true
    entail_contradiction_logits = logits[:,[0,2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    prob_label_is_true = probs[:,1]

    print(prob_label_is_true.item())

zero_shot_setting()




