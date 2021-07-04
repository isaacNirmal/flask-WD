from flask import Flask
from flask import *
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F


app = Flask(__name__)
app.secret_key = "abc"  


class_names = ['fake', 'real']
model_path = os.path.abspath('model_save')
print(model_path)

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


class ModelLoad:

    def load_model(self):
        class_names = ['fake', 'real']
        tokenizer = BertTokenizer.from_pretrained(model_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = SentimentClassifier(len(class_names))
        model = model.to(device)

        return (model, tokenizer, device)


model, tokenizer, device = ModelLoad().load_model()

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index_final.html')


@app.route("/decision",methods=["POST","GET"])
def decision():
    print(request)
    if request.method == "POST":
        subject = request.form.get("subject")
        print(subject)
        error = "invalid password"  

        

    return render_template('index_final.html',error=error) 

@app.route("/home",methods=["POST","GET"])
def home():
    print(request)
    if request.method == "POST":
        first_name = request.form.get("firstName")
        last_name = request.form.get("lastName")
        country = request.form.get("country")
        subject = request.form.get("subject")

        print(first_name,last_name,country,subject)

    return "success"

@app.route('/login',methods = ["GET","POST"])  
def login():  
    model_check = 0
    if request.method == "POST":
        subject = request.form.get("subject")
        #check the model

        encoding = tokenizer.encode_plus(
            str(subject),
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

        print(f'Tweet text: {subject}')
        print(f'Class  : {class_names[prediction]}')
        if class_names[prediction] == 'real':

            model_check = 1

        if  model_check == 0:
           
            data = {'error':'Negative'}
            return jsonify(data)

        if  model_check == 1 :
       
            data = {'error':'Positive'}
            return jsonify(data)
        

        

if __name__ == '__main__':

    
   app.run(host ='localhost', port = 5555, debug=True)