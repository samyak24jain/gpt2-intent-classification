# GPT2 causal language modeling and intent classification.

This repository has the code for the following tasks:
1. Prepare WikiText-2 dataset for causal language modeling.
2. Pre-train GPT-2 on WikiText-2 dataset for the causal language modeling task.
3. Fine-tune the model for intent classification on movie utterances - core relations dataset.

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
