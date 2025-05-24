# hackathon.py
import os
import torch, torch.nn as nn, lightning as L, torchmetrics as tm
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from pathlib import Path
from lightning.pytorch.loggers import CSVLogger

def main():
    # ---------- 1. Wczytanie --------------------------------------------
    df = pd.read_csv(Path(r"C:\Users\anton\OneDrive\Pulpit\Mastercard\Master\merged.csv"))
    cat_cols = ["merchant_id","channel","device","payment_method","currency","country.x","has_fraud_history"]
    num_cols = [c for c in df.select_dtypes(["int64","float64"]).columns if c!="is_fraud" and c not in cat_cols]

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[cat_cols] = enc.fit_transform(df[cat_cols])

    X_num = df[num_cols].astype("float32").values
    X_cat = df[cat_cols].astype("int64").values
    y     = df["is_fraud"].astype("float32").values

    Xn_tr,Xn_val,Xc_tr,Xc_val,y_tr,y_val = train_test_split(
        X_num, X_cat, y, test_size=0.2, stratify=y, random_state=42)

    Xn_tr = StandardScaler().fit_transform(Xn_tr)
    Xn_val= StandardScaler().fit_transform(Xn_val)

    train_ds = TensorDataset(torch.tensor(Xn_tr), torch.tensor(Xc_tr), torch.tensor(y_tr))
    val_ds   = TensorDataset(torch.tensor(Xn_val), torch.tensor(Xc_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=2048, num_workers=0)

    # ---------- 2. Model -------------------------------------------------
    embed_sizes = [(int(df[c].max())+2, min(50,(int(df[c].max())+3)//2)) for c in cat_cols]

    class FraudDetectorCat(L.LightningModule):
        def __init__(self, num_dim, embed_sizes, pos_weight):
            super().__init__()
            self.embeds = nn.ModuleList([nn.Embedding(n, d) for n,d in embed_sizes])
            self.mlp = nn.Sequential(
                nn.Linear(num_dim+sum(d for _,d in embed_sizes),256),
                nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128,1),
            )
            self.crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
            self.val_auc = tm.classification.BinaryAUROC()
            self.val_ap  = tm.classification.BinaryAveragePrecision()

        def forward(self,num_x,cat_x):
            emb = [e(cat_x[:,i]) for i,e in enumerate(self.embeds)]
            x   = torch.cat([num_x]+emb,1)
            return self.mlp(x).squeeze(1)

        def _step(self,batch):
            num_x, cat_x, y = batch
            logits = self(num_x,cat_x)
            return self.crit(logits,y), torch.sigmoid(logits), y

        def training_step(self,batch,_):
            loss,_,_ = self._step(batch)
            self.log("train_loss",loss)
            return loss

        def validation_step(self,batch,_):
            loss, p, y = self._step(batch)
            self.val_auc.update(p, y.long())
            self.val_ap.update(p, y.long())
            self.log("val_loss",loss,prog_bar=True)

        def on_validation_epoch_end(self):
            self.log("val_auc", self.val_auc.compute(), prog_bar=True)
            self.log("val_ap",  self.val_ap.compute(),  prog_bar=True)
            self.val_auc.reset(); self.val_ap.reset()

        def configure_optimizers(self):
            opt = torch.optim.AdamW(self.parameters(),lr=1e-3)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=10)
            return [opt],[sch]

    pos_weight = (y_tr==0).sum() / (y_tr==1).sum()
    model = FraudDetectorCat(len(num_cols), embed_sizes, pos_weight)

    # ---------- 3. Trening ----------------------------------------------
    logger = CSVLogger("lightning_logs", name="run_clean", version=None)
    trainer = L.Trainer(
        max_epochs=15,
        accelerator="cpu",
        logger=logger,
        callbacks=[L.pytorch.callbacks.EarlyStopping("val_auc", mode="max", patience=3)],
        log_every_n_steps=20
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    # Windows → potrzebny guard, żeby DataLoader z workerami >0 nie zwiesił procesu
    main()
