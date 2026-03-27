import numpy as np
from nltk.tokenize import MWETokenizer
import torch
from torch import nn
import pandas as pd
import os
from sklearn.decomposition import PCA
import timeit
import scipy

data_path = './data'
checkpoint_dir = './models/'
os.makedirs(checkpoint_dir, exist_ok=True)

class Tokenization(MWETokenizer):

    def __init__(self):
        MWETokenizer.__init__(self)
        vocab_file=open(data_path +'/drug_codes_chembl_freq_1500.txt','r')
        vocab_data=vocab_file.read()
        vocab_data=vocab_data.replace(' ','')
        self.vocab=vocab_data.split('\n')
        self.vocab.pop(0)
        self.max_length= len(max(self.vocab, key=len))
        self.zeta=58 # Max lengt of drug representation

        SMILES_CHARS = [' ','#', '%', '(', ')', '+', '-', '.', '/','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','=',
                        '@','A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P','R', 'S', 'T', 'V', 'X', 'Z',
                        '[', '\\', ']','a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's','t', 'u']
        self.vocab=self.vocab+SMILES_CHARS

        self.l=len(self.vocab)

        self.smi2index = dict( (c,i) for i,c in enumerate( self.vocab ) )
        self.index2smi = dict( (i,c) for i,c in enumerate( self.vocab ) )

        #NLTK:
        self.nltk_vocab=[]
        for token in self.vocab:
            self.nltk_vocab.append(tuple([*token]))
        self.NLTK_tokenizer = MWETokenizer(self.nltk_vocab,separator='')

    # tokenization function using NLTK package:
    def smiles_to_token(self,smiles):
        tokenized_lst=self.NLTK_tokenizer.tokenize([*smiles])
        X = np.zeros( (self.l, self.zeta ) )
        for j in range(len(tokenized_lst)):
            i=self.smi2index[tokenized_lst[j]]
            X[i,j]=1
        return X

    def token_to_smiles(self,X ):
        smi = ''
        X = X.argmax( axis=0 )
        for i in X:
            if(i==0): break
            smi += index2smi[ i ]
        return smi

    # Function to tokenize all drugs in the SMILES file and save them in a token.pkl file in data_path. This file is needed in the custom dataset.
    # Needs to be run once to create this file.
    def tokenize_file(self):
        drug_data=pd.read_csv(data_path + '/SMILEinchi.csv')
        smiles_lst=list(drug_data['smiles'])
        drugID_lst=list(drug_data['drug_id'])
        token_lst=[]
        for entry in smiles_lst:
            token_lst.append(self.smiles_to_token(entry))
        output=pd.DataFrame(list(zip(drugID_lst,token_lst)),columns=['drugID','token'])
        output=output.T
        output=output.drop(['drugID'])
        output=output.set_axis(drugID_lst,axis=1)
        output= output.loc[:,~output.columns.duplicated()].copy()
        output.to_pickle(data_path +'/token.pkl')
        return output

tok=Tokenization()
tok.tokenize_file()

class Embedding(nn.Module):

    def __init__(self,l,zeta,gamma):
        nn.Module.__init__(self)

        self.l=l # Input dim of embedding/lenght of tokenized substructure vector. Nedds to match output of Tokenization
        self.zeta=zeta # Max lengt of drug representation. Needs to match with output of Tokenization
        self.gamma=gamma # Output dim of embedding. Needs to match input of Transformer layer and be divisible by 8

        dropout_rate=0.05

        # Layers:
        self.chem_embedding=torch.nn.Linear(self.l,self.gamma,bias=False)
        self.pos_embedding=torch.nn.Linear(self.zeta,self.gamma,bias=False)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self,M):
        for i in range(M.size(0)):
            C=torch.zeros([self.gamma,self.zeta])
            P=torch.zeros([self.gamma,self.zeta])
            for j in range(self.zeta):
                C[:,j]=self.chem_embedding(M[i,:,j])
                I=torch.zeros(self.zeta)
                I=I.to(M.device)
                I[j]=1
                P[:,j]=self.pos_embedding(I)
            E_batch=torch.add(C,P)
            if i==0:
                E=E_batch.unsqueeze(0)
            else:
                E_batch=E_batch.unsqueeze(0)
                E=torch.cat((E,E_batch),0)
        E=self.dropout(E)
        return (E.to(M.device))


class Transformer(nn.Module):

    def __init__(self,gamma):
        nn.Module.__init__(self)

        encoder_layer = nn.TransformerEncoderLayer(d_model=gamma, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self,E):
        output=self.transformer_encoder(E.permute(0,2,1))
        return output

class CombinedDataset(torch.utils.data.Dataset):

    def __init__(self,gene_pca=False):

        # loading files
        self.gene_data=pd.read_table(data_path + '/Cell_line_RMA_proc_basalExp.txt')
        GDSC_data=pd.read_excel(data_path + '/GDSC2_fitted_dose_response_25Feb20.xlsx')
        self.drug_data=pd.read_pickle(data_path +'/token.pkl')

        # reduce size of gene_data file
        self.gene_names=list(self.gene_data.columns)
        self.gene_names.pop(0)
        self.gene_names.pop(0)
        self.gene_data=self.gene_data.T
        self.gene_data=self.gene_data.drop(['GENE_SYMBOLS','GENE_title'])
        self.gene_data=self.gene_data.astype('float32')

        if gene_pca:
            pca = PCA(n_components=1018)
            pca.fit(self.gene_data)
            self.gene_data=pca.transform(self.gene_data)

        self.gene_data=self.gene_data.T
        self.gene_data=pd.DataFrame(self.gene_data)
        self.gene_data=self.gene_data.set_axis(self.gene_names,axis=1)

        COSMIC_lst0= list(GDSC_data['COSMIC_ID'])
        drugID_lst0= list(GDSC_data['DRUG_ID'])
        IC50_lst0= list(GDSC_data['LN_IC50'])
        self.COSMIC_lst=[]
        self.drugID_lst=[]
        self.IC50_lst=[]

        # dropping entries with no corresponding gene/drug data
        n_fails=0
        for i in range(len(COSMIC_lst0)):
            try:
                self.gene_data['DATA.'+str(COSMIC_lst0[i])]
                self.drug_data[drugID_lst0[i]]

                self.COSMIC_lst.append(COSMIC_lst0[i])
                self.drugID_lst.append(drugID_lst0[i])
                self.IC50_lst.append(IC50_lst0[i])
            except: n_fails+=1

    def __len__(self):
        return(len(self.IC50_lst))

    def __getitem__(self,i):
        gene_expression=torch.Tensor(list(self.gene_data['DATA.'+str(self.COSMIC_lst[i])]))
        drug_token=torch.Tensor(self.drug_data.loc['token',self.drugID_lst[i]])
        IC50_value=torch.Tensor([self.IC50_lst[i]])
        return gene_expression, drug_token, IC50_value

# gene_pca reduces rna input dim from 17737 to 1018, somewhat faster. Layers in the model have to be adjusted if changed.
dat=CombinedDataset(gene_pca=True)

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.epoch=0
        self.device='cpu'
        self.emb=Embedding(2713,58,8)
        self.trafo=Transformer(8)
        self.dropout=nn.Dropout()
        self.relu=torch.nn.ReLU()
        self.lin_1=nn.Linear(1018,256)
        self.bn1 = nn.BatchNorm1d(256)
        self.lin_2=nn.Linear(256,64)
        self.bn2 = nn.BatchNorm1d(64)
        self.class_1=nn.Linear(528,64)
        self.bn3 = nn.BatchNorm1d(64)
        self.class_2=nn.Linear(64,8)
        self.bn4 = nn.BatchNorm1d(8)
        self.class_3=nn.Linear(8,1)
        self.optimizer=torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self,rna,drug):
        # rna linear layers
        rna=self.lin_1(rna)
          #rna=self.bn1(rna)
          #rna=self.relu(rna)
        rna=self.lin_2(rna)
          #rna=self.bn2(rna)
          #rna=self.relu(rna)
        # drug embedding and transformer layers
        drug=self.emb(drug)
        drug=self.trafo(drug)
        drug=torch.flatten(drug,1)
        # classifier layer
        x=torch.cat((rna,drug),dim=1)
        x=self.class_1(x)
          #x=self.bn3(x)
          #x=self.relu(x)
        x=self.class_2(x)
          #x=self.bn4(x)
          #x=self.relu(x)
        output=self.class_3(x)
        return output

    def save(self, path):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, path)

    def load(self, path):
        checkpoint = torch.load(path,map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch=checkpoint['epoch']


net=Model()

# Moving the model and the dataset to GPU, if available
def to_gpu(model):
  if torch.cuda.is_available():
      device = torch.device('cuda')  # GPU device
  else:
      device = torch.device('cpu')   # CPU device
  model.to(device)
  model.device=device
  print('device:', device)

  # Moving loaded optimizer parameters to GPU
  for state in model.optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

to_gpu(net)

# Function to move tensors in each batch of the dataloader to the GPU
def custom_collate(batch):
    if torch.cuda.is_available():
      device = torch.device('cuda')
    else:
      device = torch.device('cpu')
    gene, drug, target = zip(*batch)
    gene = torch.stack(gene).to(device)
    drug = torch.stack(drug).to(device)
    target = torch.stack(target).to(device)
    return gene, drug, target

# split dataset and define dataloader
train_set, test_set=torch.utils.data.random_split(dat, [int(0.9*dat.__len__()), dat.__len__()-int(0.9*dat.__len__())])
trainloader=torch.utils.data.DataLoader(train_set,batch_size=128,shuffle=True, collate_fn=custom_collate)
testloader=torch.utils.data.DataLoader(test_set,batch_size=128, collate_fn=custom_collate)

import torch.optim
import os
import glob
import re
import itertools # Import itertools for islice

# Training function. Running will overwrite existing model weights file!
def train(model, dataloader, n_epochs, last_epoch=0, last_batch_in_epoch=-1):
    start_time = timeit.default_timer()
    loss_lst = []
    optimizer = model.optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, threshold=0.01)

    # Only run test if truly starting from scratch (no previous checkpoint loaded)
    if last_batch_in_epoch == -1 and last_epoch == 0:
        epoch_loss, correlation = test(model, testloader)
        loss_lst.append(epoch_loss)
        time = timeit.default_timer()
        print('epochs:', model.epoch, 'test loss:', round(epoch_loss, 5), 'correlation:', round(correlation, 4), 'time:', int((time - start_time) / 60), 'min')

    # The outer loop runs for n_epochs *additional* epochs
    # model.epoch holds the absolute epoch number, already updated by net.load if resuming.

    for i in range(n_epochs):
        model.train()

        # Determine if we need to skip batches in the first epoch of this train call
        start_batch_idx_in_dataloader = 0
        if i == 0 and last_batch_in_epoch != -1:
            # If we are in the first epoch of this 'train' call and we resumed mid-epoch,
            # we need to skip batches until after the last processed batch.
            start_batch_idx_in_dataloader = last_batch_in_epoch + 1
            print(f"Resuming epoch {model.epoch} from batch {start_batch_idx_in_dataloader}")

        # Use itertools.islice to skip batches if resuming mid-epoch
        batches_iterator = enumerate(itertools.islice(dataloader, start_batch_idx_in_dataloader, None))

        for batch_id_in_dataloader, (rna, drug, target) in batches_iterator:
            # Calculate the true batch_id for checkpointing (absolute batch within the current epoch)
            current_batch_id_for_checkpoint = start_batch_idx_in_dataloader + batch_id_in_dataloader

            optimizer.zero_grad()
            output = model(rna, drug)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            if current_batch_id_for_checkpoint % 100 == 0:
                print('batch:', current_batch_id_for_checkpoint, 'train loss:', round(loss.item(), 3))
                # creating checkpoints by saving model weights once per 100 batches:
                model.save(f'./models/model1_checkpoint_batch_{model.epoch}_{current_batch_id_for_checkpoint}')

        # After completing an epoch, reset last_batch_in_epoch
        last_batch_in_epoch = -1 # Indicate that this epoch is now fully processed
        model.epoch += 1
        epoch_loss, correlation = test(model, testloader)
        scheduler.step(epoch_loss)
        loss_lst.append(epoch_loss)
        time = timeit.default_timer()
        print('epochs:', model.epoch, 'test loss:', round(epoch_loss, 5), 'correlation:', round(correlation, 4), 'time:', int((time - start_time) / 60), 'min')

# test function to evaluate model performance on test set
def test(model,dataloader):
  model.eval()
  avg_loss=0
  avg_correlation=0

  with torch.no_grad():
    for batch_id, (rna,drug,target) in enumerate(dataloader):
      output=model(rna,drug)
      loss=torch.nn.functional.mse_loss(output,target).item()
      correlation=scipy.stats.pearsonr(np.squeeze(output.cpu().numpy()), np.squeeze(target.cpu().numpy()))[0]
      avg_loss=(loss+batch_id*avg_loss)/(batch_id+1)
      avg_correlation=(correlation+batch_id*avg_correlation)/(batch_id+1)
      #if batch_id==10: break
  return avg_loss, avg_correlation

# Checkpoint loading logic
list_of_files = glob.glob(os.path.join(checkpoint_dir, 'model1_checkpoint_batch_*'))
last_epoch_loaded = 0
last_batch_in_epoch_loaded = -1 # Use -1 to indicate no batch was processed in the last epoch

if list_of_files:
    def parse_checkpoint_name(filename):
        match = re.search(r'model1_checkpoint_batch_(\d+)_(\d+)', os.path.basename(filename))
        if match:
            return int(match.group(1)), int(match.group(2)) # epoch, batch
        return -1, -1 # Invalid format for sorting

    list_of_files.sort(key=parse_checkpoint_name, reverse=True)
    latest_checkpoint = list_of_files[0]
    print(f"Loading latest checkpoint: {latest_checkpoint}")
    net.load(latest_checkpoint)

    # Extract last_epoch and last_batch_in_epoch from the loaded checkpoint name
    match = re.search(r'model1_checkpoint_batch_(\d+)_(\d+)', os.path.basename(latest_checkpoint))
    if match:
        last_epoch_loaded = int(match.group(1))
        last_batch_in_epoch_loaded = int(match.group(2))
else:
    print("No checkpoints found. Starting training from scratch.")

if __name__ == "__main__":
    train(net, trainloader, 2, last_epoch=last_epoch_loaded, last_batch_in_epoch=last_batch_in_epoch_loaded)
