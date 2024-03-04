import pandas as pd
import torch
from argparse import ArgumentParser
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Config, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, DataCollatorWithPadding
import math
import os
import evaluate
import numpy as np
import torch

# Model parameters
MAX_SEQ_LENGTH = 128
NUM_LAYERS = 6
NUM_HEADS = 8

# Pre-training parameters
PT_BATCH_SIZE = 64
PT_EVAL_BATCH_SIZE = 32
PT_EPOCHS = 1
PT_LEARNING_RATE = 2e-3
PT_WARMUP_RATIO = 0.03
PT_WEIGHT_DECAY = 0.01
PT_OUTPUT_DIR = './pretrained_gpt2_128'
PT_LOGGING_STEPS = 20
# PT_PUSH_TO_HUB = False

# Fine-tuning parameters
FT_BATCH_SIZE = 64
FT_EVAL_BATCH_SIZE = 32
FT_EPOCHS = 1
FT_LEARNING_RATE = 1e-4
FT_WARMUP_RATIO = 0.03
FT_WEIGHT_DECAY = 0.01
FT_LOGGING_STEPS = 20

def tokenize_wikitext(examples, tokenizer):
        
    concatenated_str = ''
    for x in examples['text']:
        concatenated_str += x
        concatenated_str += tokenizer.bos_token
        
    return tokenizer(concatenated_str)

def group_wikitext(examples, max_seq_len):
    
    # Get total length of tokens
    total_length = len(examples["input_ids"])

    # We drop the remainder (insignificant)
    if total_length >= max_seq_len:
        total_length = (total_length // max_seq_len) * max_seq_len
    
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + max_seq_len] for i in range(0, total_length, max_seq_len)]
        for k, t in examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    
    return result


def preprocess_core_relations(examples, tokenizer, classes, class2id):
    
    tokenized = tokenizer(examples['utterances'], truncation=True)
    
    if examples['Core Relations']:
        all_labels = examples['Core Relations'].split()
    else:
        all_labels = []
    
    labels = [0. for i in range(len(classes))]
    
    for label in all_labels:
        labels[class2id[label]] = 1.
    
    tokenized['labels'] = labels
    
    return tokenized

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def compute_ft_metrics(eval_pred):

    # Metric computation functions
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])        
    predictions, labels = eval_pred
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))
    

def pretrain():
    
    print('*'*50)
    print('PRE-TRAINING GPT2 MODEL')
    print('*'*50)
    
    print('Loading dataset from huggingface: wikitext-2-raw-v1')
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        
    print('Loading GPT2 tokenizer')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    print('Preprocessing dataset')    
    tokenized_ds = dataset.map(tokenize_wikitext, num_proc=4, batched=True, batch_size=-1, remove_columns=['text'], fn_kwargs={'tokenizer': tokenizer})
    lm_dataset = tokenized_ds.map(group_wikitext, num_proc=4, batched=True, batch_size=-1, fn_kwargs={'max_seq_len': MAX_SEQ_LENGTH})
    
    config = GPT2Config(
        n_positions=MAX_SEQ_LENGTH,
        n_layer=NUM_LAYERS,
        n_head=NUM_HEADS,
    )
    
    print('Initializing GPT2 model')
    model = AutoModelForCausalLM.from_config(config)
    
    training_args = TrainingArguments(
        output_dir=PT_OUTPUT_DIR,
        num_train_epochs=PT_EPOCHS,
        per_device_train_batch_size=PT_BATCH_SIZE,
        per_device_eval_batch_size=PT_EVAL_BATCH_SIZE,
        warmup_ratio=PT_WARMUP_RATIO,
        weight_decay=PT_WEIGHT_DECAY,
        logging_steps=PT_LOGGING_STEPS,
        logging_strategy='steps',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=PT_LEARNING_RATE,
        # push_to_hub=PT_PUSH_TO_HUB,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False,
        save_total_limit=1
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset['train'],
        eval_dataset=lm_dataset['validation']
    )
    

    print('STARTING PRETRAINING...\n')
    
    print('Pre-trained model will be saved to: ', PT_OUTPUT_DIR)
    
    print('Model parameters:')
    print(f'Batch size: {PT_BATCH_SIZE}')
    print(f'Epochs: {PT_EPOCHS}')
    print(f'Learning rate: {PT_LEARNING_RATE}')
    print(f'Warmup ratio: {PT_WARMUP_RATIO}')
    print(f'Weight decay: {PT_WEIGHT_DECAY}')
    print(f'Logging steps: {PT_LOGGING_STEPS}')
    
    trainer.train()
    
    model.save_pretrained(PT_OUTPUT_DIR)
    tokenizer.save_pretrained(PT_OUTPUT_DIR)
    
    torch.cuda.empty_cache()
    
    test_results = trainer.evaluate(eval_dataset=lm_dataset['train'])
    print(f"\n\nTRAIN PERPLEXITY = {math.exp(test_results['eval_loss']):.2f}")
    
    test_results = trainer.evaluate(eval_dataset=lm_dataset['validation'])
    print(f"\n\nVALIDATION PERPLEXITY = {math.exp(test_results['eval_loss']):.2f}")
    
    test_results = trainer.evaluate(eval_dataset=lm_dataset['test'])
    print(f"\n\nTEST PERPLEXITY = {math.exp(test_results['eval_loss']):.2f}")
    
    torch.cuda.empty_cache()

def finetune(data_path, save_model_path):
    
    print('*'*50)
    print('FINE-TUNING GPT2 MODEL')
    print('*'*50)
    
    
    if not os.path.exists(data_path):
        print('Data file does not exist')
        return
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load data and preprocess it
    dataset = load_dataset('csv', data_files=data_path)
    
    dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
    
    # rename test split to validation
    dataset['validation'] = dataset['test']
    del dataset['test']
    
    # Create class-id dictionaries for utility    
    classes = set()
    for rels in dataset['train']['Core Relations']:
        if rels:
            classes.update(rels.split())
            
    classes = list(classes)

    class2id = {class_:id for id, class_ in enumerate(classes)}
    id2class = {id:class_ for class_, id in class2id.items()}

    # Preprocess dataset
    ft_dataset = dataset.map(preprocess_core_relations, num_proc=4, batched=False, remove_columns=dataset['train'].column_names, fn_kwargs={'tokenizer': tokenizer, 'classes': classes, 'class2id': class2id})

    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    
    # Load pre-trained model - try local dir first, then huggingface
    if os.path.exists(PT_OUTPUT_DIR):
        print('Loading pre-trained model from local dir: ', PT_OUTPUT_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(PT_OUTPUT_DIR, num_labels=len(classes), id2label=id2class, label2id=class2id, problem_type = "multi_label_classification")
    else:
        print('Loading pre-trained model from Huggingface: https://huggingface.co/samyak24jain/pretrained_gpt2_128')
        model = AutoModelForSequenceClassification.from_pretrained('samyak24jain/pretrained_gpt2_128', num_labels=len(classes), id2label=id2class, label2id=class2id, problem_type = "multi_label_classification")
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token = tokenizer.pad_token
        
    # Training arguments
    training_args = TrainingArguments(
        output_dir=save_model_path,
        learning_rate=FT_LEARNING_RATE,
        per_device_train_batch_size=FT_BATCH_SIZE,
        per_device_eval_batch_size=FT_EVAL_BATCH_SIZE,
        num_train_epochs=FT_EPOCHS,
        weight_decay=FT_WEIGHT_DECAY,
        logging_steps=FT_LOGGING_STEPS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ft_dataset["train"],
        eval_dataset=ft_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_ft_metrics,
    )

    trainer.train()
    
    # Save model
    model.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_model_path)
    
    # Print final evaluation metrics
    eval_metrics = trainer.evaluate()
    
    print(eval_metrics)
        
    print('Pretrained + Finetuned GPT2 Model saved to: ', save_model_path)
    
    torch.cuda.empty_cache()
    

def train(data_path, save_model_path):
    
    pretrain()
    finetune(data_path, save_model_path)
    
    
def test(data, model_path, output_csv):
    
    print('*'*50)
    print('TESTING GPT2 MODEL')
    print('*'*50)
    
    # Read test data
    test_dataset = load_dataset('csv', data_files=data)
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    # Tokenize and preprocess test data
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    predictions = {'utterances': [], 'Core Relations': []}
    
    id2class = model.config.id2label
    
    # For each utterance, predict the core relations and convert to string format
    for i in range(len(test_dataset['train'])):
        utterance = test_dataset['train']['utterances'][i]
        tokenized = tokenizer(utterance, truncation=True, padding='max_length', max_length=MAX_SEQ_LENGTH, return_tensors='pt').to(device)
        output = model(**tokenized)
        logits = sigmoid(output.logits[0].detach().cpu().numpy())
        predictions['utterances'].append(utterance)
        predictions['Core Relations'].append(' '.join([id2class[i] for i in range(len(logits)) if logits[i] > 0.5]))
    
    # Create DataFrame from the predictions
    output_df = pd.DataFrame(predictions)
    
    # # Save predictions to CSV
    output_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

def main():
    parser = ArgumentParser("NLP 244 HW2 CLI")

    parser.add_argument('--train', action="store_true", help="indicator to train model")
    parser.add_argument('--test', action="store_true", help="indicator to test model")

    parser.add_argument('--data', help="path to data file")
    parser.add_argument('--save_model', help="ouput path of trained model")
    parser.add_argument('--model_path', help="path to load trained model from")

    parser.add_argument('--output', help="output path of predictions")

    args = parser.parse_args()

    if args.train:
        train(args.data, args.save_model)
    if args.test:
        test(args.data, args.model_path, args.output)
        

if __name__ == "__main__":
    main()