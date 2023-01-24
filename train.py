from utils import *
from transformers import AutoModel,AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer



import torch.optim as optim

model = AutoModel.from_pretrained("microsoft/deberta-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

train_loader = DataLoader(Data(tokenizer,"training"),batch_size=16,collate_fn=collate_fn,shuffle=True)
val_loader = DataLoader(Data(tokenizer,"validation"),batch_size=16,collate_fn=collate_fn)


wandb_logger = WandbLogger(project="values")



# define the LightningModule
class MLClassifier(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model=model
        self.classifier = nn.Sequential(*[nn.Linear(768,128),
                                         nn.Linear(128,20)])
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self,x):
        emb = self.model(x)['last_hidden_state'][:,0,:]
        o = self.classifier(emb)
        return o

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        X,y = batch
        o = self.forward(X)
        loss = self.loss_fn(o, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss


    def validation_step(self, batch,batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        X,y = batch
        o = self.forward(X)
        loss = self.loss_fn(o, y)
        # Logging to TensorBoard (if installed) by default
        self.log("Validation loss", loss,on_epoch=True)
    
    def predict_step(self, batch, batch_idx):
        # take average of `self.mc_iteration` iterations
        X,y = batch
        pred = self(X)
        return pred,y

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer




# init the autoencoder
clf = MLClassifier(model)

trainer = pl.Trainer( max_epochs=1,accelerator="gpu", devices=1,logger=wandb_logger)
trainer.fit(model=clf, train_dataloaders=train_loader,val_dataloaders = val_loader)

predictions = trainer.predict(model=clf,dataloaders=train_loader)
print(predictions)