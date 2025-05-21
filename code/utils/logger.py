from loguru import logger


def setup_logger():
    logger.remove()
    logger.add(
        sink="stdout",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO"
    )
    return logger
```

---

## utils/metrics.py
```python
import torch
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_metrics(preds, targets):
    preds = preds.numpy()
    targets = targets.numpy()
    return {
        'precision': precision_score(targets, preds, average='macro'),
        'recall':    recall_score(targets, preds, average='macro'),
        'f1':        f1_score(targets, preds, average='weighted')
    }