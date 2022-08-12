
from classification_models.lightCNN import litCNN
from classification_models.lightSimpleRNN import litSimpleRNN
from classification_models.lightTT import litTNLM
from utils.cls_data_process import vocab, loader_train,loader_validation
import wandb

# generate data batch and iterator
from pytorch_lightning.loggers import WandbLogger
batch_size = 64

import pytorch_lightning as pl

# wandb_logger = WandbLogger(name="TNLMAAAI", project="text_classification")

model = litCNN(len(vocab), e_dim=300, h_dim=64, o_dim=2)

# model = litRNN(
#     vocab_size=len(vocab),
#     embedding_dim=300,
#     hidden_dim=100,
#     num_classes=2,
#     lstm_layers=2,
#     bidirectional=True,
#     dropout=0.5,
#     pad_index=0,
# )

# model = litSimpleRNN(
#     vocab_size=len(vocab), embed_size=300, hidden_dim=256, output_size=2
# )
# model = litTNLM(rank=5, vocab_size=len(vocab), output_size=2)
# wandb_logger.watch(model, log="all")
trainer = pl.Trainer(logger=None, max_epochs=20, accelerator="gpu")
trainer.fit(model, loader_train, loader_validation)
# wandb.finish()

# %%
