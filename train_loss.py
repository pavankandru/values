import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger(project="values")

class1=['Be creative', 'Be curious', 'Have freedom of thought',
       'Be choosing own goals', 'Be independent', 'Have freedom of action',
       'Have privacy', 'Have an exciting life', 'Have a varied life',
       'Be daring', 'Have pleasure', 'Be ambitious', 'Have success',
       'Be capable', 'Be intellectual', 'Be courageous', 'Have influence',
       'Have the right to command', 'Have wealth', 'Have social recognition',
       'Have a good reputation', 'Have a sense of belonging',
       'Have good health', 'Have no debts', 'Be neat and tidy',
       'Have a comfortable life', 'Have a safe country',
       'Have a stable society', 'Be respecting traditions',
       'Be holding religious faith', 'Be compliant', 'Be self-disciplined',
       'Be behaving properly', 'Be polite', 'Be honoring elders', 'Be humble',
       'Have life accepted as is', 'Be helpful', 'Be honest', 'Be forgiving',
       'Have the own family secured', 'Be loving', 'Be responsible',
       'Have loyalty towards friends', 'Have equality', 'Be just',
       'Have a world at peace', 'Be protecting the environment',
       'Have harmony with nature', 'Have a world of beauty', 'Be broadminded',
       'Have the wisdom to accept others', 'Be logical',
       'Have an objective view']

class2=['Self-direction: thought', 'Self-direction: action',
       'Stimulation', 'Hedonism', 'Achievement', 'Power: dominance',
       'Power: resources', 'Face', 'Security: personal', 'Security: societal',
       'Tradition', 'Conformity: rules', 'Conformity: interpersonal',
       'Humility', 'Benevolence: caring', 'Benevolence: dependability',
       'Universalism: concern', 'Universalism: nature',
       'Universalism: tolerance', 'Universalism: objectivity']

class2b=['Self-direction', 'Power', 'Security', 'Conformity', 'Benevolence','Universalism']

class3=['Openness to change', 'Self-enhancement', 'Conservation',
       'Self-transcendence']

class4 = ['Personal focus', 'Social focus']
class4b = ['Growth, Anxiety-free','Self-protection, Anxiety avoidance']

weights={}


def update_weights(a,b,c,d,e,f):
    global weights
    weights=[]
    lens = np.array([i.shape[1] for i in [a,b,c,d,e,f]])
    cat_weigths = 10*lens/lens.sum()
    ans_=[]
    for i,j in zip([a,b,c,d,e,f],cat_weigths):
        l=i.sum(axis=0)
        x = sum(l)/l
        x = x/x.sum()
        weights.append(torch.Tensor(j*x))

    
    

def extra_labels(df,split='training'):
    df['Self-direction'] = df[['Self-direction: thought', 'Self-direction: action']].sum(axis=1)
    df['Power'] = df[['Power: dominance','Power: resources']].sum(axis=1)
    df['Security']= df[['Security: personal', 'Security: societal']].sum(axis=1)
    df['Conformity']= df[['Conformity: rules', 'Conformity: interpersonal']].sum(axis=1)
    df['Benevolence']=df[['Benevolence: caring', 'Benevolence: dependability']].sum(axis=1)
    df['Universalism'] =df[['Universalism: concern', 'Universalism: nature','Universalism: tolerance', 'Universalism: objectivity']].sum(axis=1)
    for i in ['Self-direction','Power','Security','Conformity','Benevolence','Universalism']:
        df[i]=(df[i]>0)*1

    l1 = pd.read_csv('data/level1-labels-{}.tsv'.format(split),delimiter='\t')
    l3 = pd.read_csv('data/labels-level3.tsv',delimiter='\t')
    l4a = pd.read_csv('data/labels-level4a.tsv',delimiter='\t')
    l4b = pd.read_csv('data/labels-level4b.tsv',delimiter='\t')
    df = pd.merge(df,l1,on='Argument ID')
    df = pd.merge(df,l3,on='Argument ID')
    df = pd.merge(df,l4a,on='Argument ID')
    df = pd.merge(df,l4b,on='Argument ID')

    return df

class Data(Dataset):
    def __init__(self,tokenizer,split="training") -> None:
        super().__init__()
        df = pd.read_csv('data/arguments-{}.tsv'.format(split),delimiter='\t')
        if split!='test':
            df2 = pd.read_csv('data/labels-{}.tsv'.format(split),delimiter='\t')
            df = df.merge(df2,on=["Argument ID"])
            df = extra_labels(df,split)
            self.labels2 = df[class2].values
            self.labels2b= df[class2b].values
            self.labels3= df[class3].values
            self.labels4=df[class4].values
            self.labels4b=df[class4b].values
            self.labels1=df[class1].values
        
        self.argument_ids  = df['Argument ID'].values
        self.conclusion =df['Conclusion'].values
        self.stance  = df['Stance'].values
        self.premise = df['Premise'].values
        self.tokenizer =tokenizer
        self.split=split
        if split=='training':
            update_weights(self.labels2,self.labels2b,self.labels1,self.labels3,self.labels4,self.labels4b)

    def __len__(self):
        return len(self.argument_ids)
    
    def __getitem__(self,idx):
        return_dict =  {"premise":self.tokenizer(self.premise[idx],add_special_tokens=False)['input_ids'],
                "stance":self.tokenizer(self.stance[idx],add_special_tokens=False)['input_ids'],
                "conclusion":self.tokenizer(self.conclusion[idx],add_special_tokens=False)['input_ids'],
                }
        if self.split!='test':
            return_dict["labels1"]=self.labels1[idx]
            return_dict["labels2"]=self.labels2[idx]
            return_dict["labels2b"]=self.labels2b[idx]
            return_dict["labels3"]=self.labels3[idx]
            return_dict["labels4"]=self.labels4[idx]
            return_dict["labels4b"]=self.labels4b[idx]

        return return_dict


def collate_fn(batch):
    unpadded_batch=[]
    for row in batch:
        i,j,k = row['premise'],row['stance'],row['conclusion']
        unpadded_batch.append([1]+i+[2]+j+[2]+k+[2])
    max_len = max([len(i) for i in unpadded_batch])
    x = np.array([i+[0]*(max_len-len(i)) for i in unpadded_batch])
    
    return torch.LongTensor(x),\
        (torch.Tensor(np.array([row['labels1'] for row in batch])) if 'labels1' in batch[0] else None),\
        (torch.Tensor(np.array([row['labels2'] for row in batch])) if 'labels2' in batch[0] else None),\
        (torch.Tensor(np.array([row['labels2b'] for row in batch])) if 'labels2b' in batch[0] else None),\
        (torch.Tensor(np.array([row['labels3'] for row in batch])) if 'labels3' in batch[0] else None),\
        (torch.Tensor(np.array([row['labels4'] for row in batch])) if 'labels4' in batch[0] else None),\
        (torch.Tensor(np.array([row['labels4b'] for row in batch])) if 'labels4b' in batch[0] else None)


    

from transformers import AutoModel,AutoTokenizer
import pytorch_lightning as pl


from pytorch_lightning import Trainer
from sklearn.metrics import classification_report


import pickle



import torch.optim as optim

model = AutoModel.from_pretrained("microsoft/deberta-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

train_loader = DataLoader(Data(tokenizer,"training"),batch_size=16,collate_fn=collate_fn,shuffle=True)
val_loader = DataLoader(Data(tokenizer,"validation"),batch_size=16,collate_fn=collate_fn,)
test_loader = DataLoader(Data(tokenizer,"test"),batch_size=16,collate_fn=collate_fn,)






# define the LightningModule
class MLClassifier(pl.LightningModule):
    def __init__(self, model,dropout=0.2):
        super().__init__()
        self.model=model
        
        self.classifier2=nn.Sequential(*[nn.Dropout(dropout),
                                         nn.Linear(768,128),
                                         nn.Linear(128,20)])
        self.classifier2b=nn.Sequential(*[nn.Dropout(dropout),
                                         nn.Linear(768,128),
                                         nn.Linear(128,6)])
        self.classifier1=nn.Sequential(*[nn.Dropout(dropout),
                                         nn.Linear(768,128),
                                         nn.Linear(128,54)])
        self.classifier3=nn.Sequential(*[nn.Dropout(dropout),
                                         nn.Linear(768,128),
                                         nn.Linear(128,4)])
        self.classifier4=nn.Sequential(*[nn.Dropout(dropout),
                                         nn.Linear(768,128),
                                         nn.Linear(128,2)])
        self.classifier4b=nn.Sequential(*[nn.Dropout(dropout),
                                         nn.Linear(768,128),
                                         nn.Linear(128,2)])
        self.clsfrs=[self.classifier1,self.classifier2,self.classifier2b,
                     self.classifier3,self.classifier4,self.classifier4b]
        
        self.loss_fn2 = nn.MultiLabelSoftMarginLoss(weight=weights[0])
        self.loss_fn2b = nn.MultiLabelSoftMarginLoss(weight=weights[1])
        self.loss_fn1 = nn.MultiLabelSoftMarginLoss(weight=weights[2])
        self.loss_fn3 = nn.MultiLabelSoftMarginLoss(weight=weights[3])
        self.loss_fn4 = nn.MultiLabelSoftMarginLoss(weight=weights[4])
        self.loss_fn4b = nn.MultiLabelSoftMarginLoss(weight=weights[5])
        

    def forward(self,x):
        emb = self.model(x,output_hidden_states=True)['hidden_states']
        os=[]
        for i,j in zip(self.clsfrs,[11,12,10,8,7,6]):
            os.append(i(emb[j][:,0,:]))
        # print([i.shape for i in os])
        # print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
        return os

    def prediction_reducer(self,otps):
        predictions = torch.cat([i['predictions'][1].detach() for i in otps])
        # predictions = predictions[:,[0, 2, 3, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]]
        predictions = torch.sigmoid(predictions)
        if 'labels' in otps[0]:
            labels = torch.cat([i['labels'][1].detach() for i in otps])
            # labels=labels[:,[0, 2, 3, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]]
            return predictions,labels
        return predictions

    def training_step(self, batch, batch_idx):
        X,l1,l2,l2b,l3,l4,l4b = batch
        o = self.forward(X)

        loss1 = self.loss_fn1(o[0],l1 )
        loss2 = self.loss_fn2(o[1],l2 )
        loss2b = self.loss_fn2b(o[2],l2b )
        loss3 = self.loss_fn3(o[3],l3)
        loss4 = self.loss_fn4(o[4],l4 )
        loss4b = self.loss_fn4b(o[5],l4b )
        # Logging to TensorBoard (if installed) by 
        loss = loss1+loss2+loss2b+loss3+loss4+loss4b
        self.log("train_loss",loss )
        return {"loss": loss, "predictions": o,"labels":[l1,l2,l2b,l3,l4,l4b]}

    def training_epoch_end(self, training_step_outputs):
        predictions,labels = self.prediction_reducer(training_step_outputs)
        # print("\n\n\nHere\n\n\n",predictions.shape,labels.shape)
        report = classification_report(predictions.cpu()>0.5,labels.cpu(),target_names=class2,zero_division=0,output_dict=True)
        self.log("Training Macro F1", report['macro avg']['f1-score'],on_epoch=True)
        self.log("Training Micro F1", report['micro avg']['f1-score'],on_epoch=True)
        print('\n\n Training')
        print(classification_report(predictions.cpu()>0.5,labels.cpu(),target_names=class2,zero_division=0))

    def validation_step(self, batch,batch_idx):
        X,l1,l2,l2b,l3,l4,l4b = batch
        o = self.forward(X)

        loss1 = self.loss_fn1(o[0],l1 )
        loss2 = self.loss_fn2(o[1],l2 )
        loss2b = self.loss_fn2b(o[2],l2b )
        loss3 = self.loss_fn3(o[3],l3)
        loss4 = self.loss_fn4(o[4],l4 )
        loss4b = self.loss_fn4b(o[5],l4b )
        # Logging to TensorBoard (if installed) by 
        loss = loss1+loss2+loss2b+loss3+loss4+loss4b
        self.log("Validation Loss",loss )
        return {"loss": loss, "predictions": o,"labels":[l1,l2,l2b,l3,l4,l4b]}

    def validation_epoch_end(self, training_step_outputs):
        predictions,labels = self.prediction_reducer(training_step_outputs)
        # print("\n\n\nHere\n\n\n",predictions.shape,labels.shape)
        report = classification_report(predictions.cpu()>0.5,labels.cpu(),target_names=class2,zero_division=0,output_dict=True)
        self.log("Validation Macro F1", report['macro avg']['f1-score'],on_epoch=True)
        self.log("Validation Micro F1", report['micro avg']['f1-score'],on_epoch=True)
        print('\n\n Validation')
        print(classification_report(predictions.cpu()>0.5,labels.cpu(),target_names=class2,zero_division=0))
    
    def predict_step(self, batch, batch_idx):
        # take average of `self.mc_iteration` iterations
        X,l1,l2,l2b,l3,l4,l4b = batch
        pred = self(X)
        return pred[1]
    
    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        # return optimizer
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.3)
        return {"optimizer": optimizer,  "lr_scheduler": lr_scheduler}



# init the autoencoder
clf = MLClassifier(model)

trainer = pl.Trainer( max_epochs=20,accelerator="gpu", devices=1,logger=wandb_logger,)
trainer.fit(model=clf, train_dataloaders=train_loader,val_dataloaders = val_loader)
# trainer.fit(model=clf, train_dataloaders=val_loader,val_dataloaders = val_loader)


# from temperature_scaling import ModelWithTemperature
# scaled_model = ModelWithTemperature(clf)
# scaled_model.set_temperature(val_loader)




predictions_test = trainer.predict(model=clf,dataloaders=test_loader)
predictions = trainer.predict(model=clf,dataloaders=val_loader)


pickle.dump(predictions,open('predictions_val4loss','wb'))
pickle.dump(predictions_test,open('predictions_test4loss','wb'))