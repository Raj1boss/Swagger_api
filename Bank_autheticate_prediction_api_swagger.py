from flask import Flask,request
import pandas as pd
import pickle
from flasgger import Swagger
from pdfminer.high_level import extract_text
import re
#from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import spacy
from spacy import displacy
from werkzeug.utils import secure_filename
# from PIL import Image


app=Flask(__name__)
Swagger(app)


pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)


@app.route('/')
def welcome():
    return "Welcome Home"


@app.route('/predict',methods=['Get'])
def predict_note_authentication():
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    return "The predicted values is"+ str(prediction)
    


@app.route('/predict_file',methods=['POST'])
def predict_note_file():
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df_test)
    return str(list(prediction))



#Tag Name Entity (NER)

def take_pdf(pdf):
    """This function takes pdf as paramente and return text
    """
    
    text=extract_text(pdf)
    return text


def pre_processing(text):
    text=re.sub(r'\[[0-9]*\]'," ",text)
    text=re.sub(r'\s+',' ',text)
    text=text.lower()
    text=re.sub(r'\d','',text)
    text=re.sub(r'\s+',' ',text)
    html=re.compile(r'<[^>]+>')
    text=html.sub("",text)
    text= re.sub('http://\S+|https://\S+', '', text)
    text = re.sub('http[s]?://\S+', '', text)
    text = re.sub(r"http\S+", "", text)
    return text

def convert_text(text):
    sent_tk=nltk.sent_tokenize(text)
    filtered_text=[]
    wordnet_lemmatizer = WordNetLemmatizer()
    
    for i in range(len(sent_tk)):
        word1=nltk.word_tokenize(sent_tk[i])
        words=[wordnet_lemmatizer.lemmatize(word) for word in word1 if word not in set(stopwords.words('english'))]
        filtered_text.append(words)
        
    
    text1=''
    for i in range(len(filtered_text)):
        for j in filtered_text[i]:
            text1=text1+j+" " 
        
    org_text=text1.replace(":","").replace(" '' `` ,","").replace(" ``",'').replace("''","").replace("/","").replace(",","").replace(".","")
    return org_text





@app.route('/predict_nlp_text',methods=['POST'])
def nlp_text():
    """Let's analysis the text  
    This is using docstrings for specifications.
    ---
    parameters:  
      
        
      - name: file
        in: formData
        type: file
        required: true 
     
    responses:
        200:
            description: The output values
        
    """
    request_data = request.files.get('file')
    data=secure_filename(request_data.filename)
    text1 = take_pdf(data)
    print("The TExt_2: ",text1) 
    
    text2 = pre_processing(text1)
    text3 = convert_text(text2)
    nlp=spacy.load('en_core_web_sm')
    nl_text=nlp(text3)
    
    html_text=displacy.render(nl_text,style='ent')
    ent_text=re.sub('<[^<]+?>', '', html_text) 
    return ent_text
    
    
    
    
    

if __name__=='__main__':
    app.run()