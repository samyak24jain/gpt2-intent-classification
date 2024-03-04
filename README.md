# GPT2: Causal LM and Intent Classification

This repository has the code for the following tasks:
1. Pre-training GPT-2 for the causal language modeling task.
2. Fine-tuning GPT-2 for the intent classification task.

- Pre-training dataset: WikiText-2
- Fine-tuning dataset: Movie utterances - core relations dataset.

### Requirements
```
# Ideally, do this in a new virtual environment.
pip install -r requirements.txt
```

### Training and evaluation
```
# Run main.py to train your best model on the file specified by --data
python main.py --train --data "./train.csv" --save_model "./joint_trained_model.pt"

# Run main.py to test the newly trained model on the test data
python main.py --test --data "./test.csv" --model_path "./joint_trained_model.pt" --output "./preds.csv"
```
