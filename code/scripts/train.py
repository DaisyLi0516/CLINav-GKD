import argparse
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import clip
import yaml
from pathlib import Path
from data.dataset import YourDataset
from models.student_encoder import StudentEncoder
from utils.logger import setup_logger
from utils.loss import TopoLa
import torch.nn.functional as F


def train(cfg):
    logger = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP model
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    # Prepare datasets (assumes dataset returns: image, token, idx, class_label, plac3d_label, rote6d_label)
    train_txt = Path(cfg['data_dir']) / "MC6D_train.txt"
    val_txt   = Path(cfg['data_dir']) / "MC6D_val.txt"
    train_ds  = YourDataset(cfg['data_dir'], cfg['train_label'], train_txt, preprocess)
    val_ds    = YourDataset(cfg['data_dir'], cfg['val_label'],   val_txt,   preprocess)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False)

    # Initialize student model
    student = StudentEncoder(output_dim=cfg['output_dim'], pretrained=cfg['pretrained']).to(device)
    optimizer = optim.Adam(student.parameters(), lr=cfg['learning_rate'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=0.1)

    # Task-specific losses
    loss_class  = torch.nn.CrossEntropyLoss()
    loss_plac3d = torch.nn.CrossEntropyLoss()
    loss_rote6d = torch.nn.L1Loss(reduction='mean')

    for epoch in range(cfg['epochs']):
        student.train()
        total_loss = 0.0
        for batch in train_loader:
            imgs, tokens, idxs, label_c, label_p, label_r = batch
            imgs = imgs.to(device)
            label_c = label_c.to(device)
            label_p = label_p.to(device)
            label_r = label_r.to(device)

            # Teacher features (no grad)
            with torch.no_grad():
                t_feats = clip_model.encode_image(imgs)
            # Student features
            s_feats = student(imgs)

            # Normalize
            t_feats = t_feats / t_feats.norm(dim=-1, keepdim=True)
            s_feats = s_feats / s_feats.norm(dim=-1, keepdim=True)

            # Classification logits using current features and text tokens
            logit_scale = clip_model.logit_scale.exp()
            txt_feats = clip_model.encode_text(tokens.to(device))
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            logits_per_image = (logit_scale * s_feats @ txt_feats.T)
            # Reshape and sum for multi-task
            B, C = logits_per_image.shape
            # Assuming C = num_classes * num_plac3d * num_rote6d
            # e.g., C = 7*7*7
            logits_reshaped = logits_per_image.view(B, -1, cfg['dim'], cfg['dim'])  # adjust dims accordingly
            sim_class = logits_reshaped.sum(3).sum(2)
            sim_plac  = logits_reshaped.sum(3).sum(1)
            sim_rote  = logits_reshaped.sum(2).sum(1)

            # Compute task losses
            cls_loss   = loss_class(sim_class, label_c)
            plac_loss  = loss_plac3d(sim_plac.float(), label_p)
            rote_loss  = loss_rote6d(sim_rote.float(), label_r)

            # Distillation via TopoLa
            topo_gt   = TopoLa(t_feats, t_feats, cfg['lambda_val'])
            topo_pred = TopoLa(t_feats, s_feats, cfg['lambda_val'])
            distill_loss = F.mse_loss(topo_pred, topo_gt)

            # Total loss
            loss = 0.5*cls_loss + 0.5*plac_loss + 0.5*rote_loss + 0.5*distill_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{cfg['epochs']} - Loss: {avg_loss:.4f}")
        torch.save(student.state_dict(), cfg['checkpoint'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    train(cfg)
```
---python
import argparse
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import clip
import yaml
from pathlib import Path
from data.dataset import YourDataset
from models.student_encoder import StudentEncoder
from utils.logger import setup_logger


def train(cfg):
    logger = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    train_txt = Path(cfg['data_dir']) / "MC6D_train.txt"
    val_txt   = Path(cfg['data_dir']) / "MC6D_val.txt"
    train_ds  = YourDataset(cfg['data_dir'], cfg['train_label'], train_txt, preprocess)
    val_ds    = YourDataset(cfg['data_dir'], cfg['val_label'],   val_txt,   preprocess)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False)

    student = StudentEncoder(output_dim=cfg['output_dim'], pretrained=cfg['pretrained']).to(device)
    optimizer = optim.Adam(student.parameters(), lr=cfg['learning_rate'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=0.1)

    for epoch in range(cfg['epochs']):
        student.train()
        total_loss = 0
        for imgs, tokens, _ in train_loader:
            imgs = imgs.to(device)
            with torch.no_grad():
                t_feats = clip_model.encode_image(imgs)
            s_feats = student(imgs)
            t_feats = t_feats / t_feats.norm(dim=-1, keepdim=True)
            s_feats = s_feats / s_feats.norm(dim=-1, keepdim=True)
            loss = ((t_feats - s_feats)**2).mean()

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        logger.info(f"Epoch {epoch+1}/{cfg['epochs']} - Loss: {total_loss/len(train_loader):.4f}")
        torch.save(student.state_dict(), cfg['checkpoint'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    train(cfg)