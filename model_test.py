from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F

class_names = ['fake', 'real']
model_path = os.path.abspath('model_save')
class SentimentClassifier(nn.Module):
    
  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(model_path)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)



print(os.chdir("c:/Users/misaac/Pictures/Camera Roll/Desktop/dev/main_project"))

print(os.path.abspath('model_save'))
model_path = os.path.abspath('model_save')
tokenizer = BertTokenizer.from_pretrained(model_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


sample_txt = "Take simple daily precautions to help prevent the spread of respiratory illnesses like COVID19"
model = SentimentClassifier(len(class_names))
model = model.to(device)





encoding = tokenizer.encode_plus(
  sample_txt,
  max_length=32,
  add_special_tokens=True,
  return_token_type_ids=False,
  pad_to_max_length=True,
  return_attention_mask=True,
  return_tensors='pt',
)

input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)

output = model(input_ids, attention_mask)
_, prediction = torch.max(output, dim=1)

print(f'Tweet text: {sample_txt}')
print(f'Class  : {class_names[prediction]}')