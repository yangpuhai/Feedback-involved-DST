
import pytorch_lightning as pl
from transformers import AdamW

class DST_Seq2Seq(pl.LightningModule):

    def __init__(self,args, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.lr = args["lr"]

    def training_step(self, batch, batch_idx):
        loss = self.model(input_ids=batch["encoder_input"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["decoder_output"]).loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(input_ids=batch["encoder_input"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["decoder_output"]).loss
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, correct_bias=True)
