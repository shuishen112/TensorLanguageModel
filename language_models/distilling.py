import random

import torch
import torch.nn as nn
from kd_Tool import distillation_loss
from tensor_language_model.tnlm import TN
from transformer_bert.bert_language_model import BertForMaskedLM
from transformers import AdamW, AutoConfig, AutoTokenizer

config = AutoConfig.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
print(config.vocab_size)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

with open("lm_data/clean.txt",'r') as fp:
    text = fp.read().split("\n")

print(text[:5])

inputs = tokenizer(text, return_tensors='pt', max_length=10, truncation=True, padding='max_length')
inputs


# get student label
selection = random.sample(range(1, 512), inputs.input_ids.shape[0])
selection[:5]

inputs['labels'] = torch.tensor(selection)
inputs.keys()

# get teacher label
inputs['teacher_labels'] = inputs.input_ids.detach().clone()

# mask the token, we only mask one token
for i in range(inputs.input_ids.shape[0]):
    inputs.input_ids[i, selection[i]] = 103


# load teacher model

teacher_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
teacher_model.to(device)

class MeditationsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

dataset = MeditationsDataset(inputs)

loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

model = TN(rank = 10, vocab_size = config.vocab_size)


# and move our model over to the selected device
model.to(device)
# activate training mode
model.train()



# initialize optimizer
optim = AdamW(model.parameters(), lr=5e-5)

from torch.nn import CrossEntropyLoss
from tqdm import tqdm  # for our progress bar

epochs = 2
for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        # attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        teacher_labels = batch['teacher_labels'].to(device)
        # process

        # get teacher logits
        techer_outputs = teacher_model(input_ids, attention_mask=attention_mask,
                        labels=teacher_labels)
        t_logits = torch.nn.functional.softmax(techer_outputs.logits,-1)
        t_logits_masked = t_logits[:,labels,:].squeeze(1)
        
        # get student logits
        s_logits = model(input_ids)
        logits = torch.nn.functional.softmax(s_logits)
        # print(t_logits_masked)
        # print(logits)

        merged_loss, d_loss, nll_loss = distillation_loss(logits.view(-1, model.vocab_size),labels.view(-1),t_logits_masked,output_mode = "classification")
        
        # loss_fct = CrossEntropyLoss()  # -100 index = padding token
        # masked_lm_loss = loss_fct(logits.view(-1, model.vocab_size), labels.view(-1))
        # # extract loss
        loss = merged_loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())