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
MFMA-DTI uses pretrained language models to extract semantic representations of drugs and proteins. To facilitate reproducibility, the exact pretrained checkpoints and their configurations are specified below.

#### 1. ChemBERTa for Drug Representation

- **Model:** `ChemBERTa-zinc-base-v1`
- **Hugging Face checkpoint:** `seyonec/ChemBERTa-zinc-base-v1`
- **Architecture:** RoBERTa (`RobertaForMaskedLM`)
- **Pretraining objective:** Masked Language Modeling (MLM)
- **Pretraining data:** approximately 100K chemical SMILES from the ZINC dataset
- **Hidden size:** 768
- **Number of Transformer layers:** 6
- **Number of attention heads:** 12
- **Intermediate size:** 3072
- **Vocabulary size:** 767
- **Maximum position embeddings:** 514
- **Tokenizer:** ByteLevel tokenizer
- **Output feature dimension:** 768
- **Fine-tuning:** The pretrained ChemBERTa parameters are **not fine-tuned on the DTI datasets**. The model is used as a fixed feature extractor to generate drug semantic representations.
- **Source:** https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1

The locally stored ChemBERTa configuration is consistent with the `ChemBERTa-zinc-base-v1` checkpoint. In particular, it uses a RoBERTa architecture with 6 Transformer layers, a hidden size of 768, and 12 attention heads.

#### 2. ProtBERT for Protein Representation

- **Model:** `ProtBERT`
- **Hugging Face checkpoint:** `Rostlab/prot_bert`
- **Architecture:** BERT (`BertForMaskedLM`)
- **Pretraining objective:** Masked Language Modeling (MLM)
- **Pretraining data:** UniRef100
- **Hidden size:** 1024
- **Number of Transformer layers:** 30
- **Number of attention heads:** 16
- **Intermediate size:** 4096
- **Vocabulary size:** 30
- **Maximum position embeddings:** 40,000
- **Output feature dimension:** 1024
- **Fine-tuning:** The pretrained ProtBERT parameters are **not fine-tuned on the DTI datasets**. The model is used as a fixed feature extractor to generate protein semantic representations.
- **Source:** https://huggingface.co/Rostlab/prot_bert





