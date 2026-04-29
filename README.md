## MFMA-DTI:
A Multi-Feature Fusion and Meta-Path Augmented Framework for Drug-Target Interaction Prediction

### Overview
MFMA-DTI is a heterogeneous graph-based framework for drug-target interaction (DTI) prediction. It integrates topology-aware graph learning with pretrained semantic features through a dual-stream architecture, and introduces masking augmentation and topology-semantic contrastive learning to improve generalization.

### envs

| Name         | Version      | 
|--------------|--------------|
| python       | 3.8.0        |        
| dgl          | 0.8.0+       |      
| networkx     | 3.1          |       
| numpy        | 1.24.3       |      
| pandas       | 2.0.3        |      
| scikit-learn | 1.3.0        |         
| scipy        | 1.10.1       |           
| torch        | 1.9.0+cu111  |          
| torchaudio   | 0.9.0        |          
| torchvision  | 0.10.0+cu111 |            
| tqdm         | 4x           |          

- Install dependencies:

```Python
pip install torch dgl transformers scikit-learn numpy scipy tqdm
```

### Quick start

#### Step 1 — Generate Pretrained Features
Before training, generate drug and protein embeddings using pretrained models (ChemBERTa for drugs, ProtBERT for proteins).

- Run `./generate_pretrained_features.py` to obtain the pretrained models drug and protein embeddings for use by the model.

```Python
python generate_pretrained_features.py 

```
#### Step 2 — Train the Model

Run the training script with 5-fold cross-validation:

- Run `./main.py` predict drug-target interactions. 

```Python
python main.py --epochs=200 --inp_size=256 --hidden_size=256 --out_size=256 --dropout=0.3 --learning_rate=0.001 --dataname=heter
```
We can download the source code and data using Git.

```
git clone https://github.com/HENU406/MFMA-DTI.git
```


### File Structure
The file structure is as follows: 
```
CE-DTI

└───data

│   └─── heter

│   └─── zheng

└───model.py

└───main.py

└───utils.py

└───generate_pretrained_features.py


```

### Pretrain-Model

The Pretrain model mentioned in the paper can be found in [ChemBERTa](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM) and [ProtBERT](https://huggingface.co/Rostlab/prot_bert).








