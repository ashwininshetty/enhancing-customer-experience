import gradio as gr
import pickle
import nltk
import string
import os
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import requests
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

model_URL='https://gv-smdm.s3.amazonaws.com/model.sav'
vect_url ='https://gv-smdm.s3.amazonaws.com/TFIDF_Vect.pkl'

response = requests.get(model_URL)
open('model.sav', "wb").write(response.content)

response = requests.get(vect_url)
open('TFIDF_Vect.pkl', "wb").write(response.content)

model = pickle.load(open('model.sav','rb'))
TFIDF_Vect = pickle.load(open('TFIDF_Vect.pkl','rb'))

# Define stopwords, punctuations, and special characters to remove from the text
stopwords_punctuations = set(stopwords.words('english')).union(set(string.punctuation)).union(['--', 'xxxx', "''", '""', '...', '``'])

# Initialize WordNetLemmatizer for lemmatizing the words
lemmatizer = WordNetLemmatizer()

# Function to process text data
def process_text(text):
    # Tokenize the text data
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords, punctuations, and special characters
    sw_punct_rmd = [token.lower() for token in tokens if token.lower() not in stopwords_punctuations and token.isalpha()]
    
    # Lemmatize the words
    lemmatized_words = [lemmatizer.lemmatize(word) for word in sw_punct_rmd]
    
    # Join the words into a single string and return
    processed_text = ' '.join(lemmatized_words)
    return processed_text

def predict(narrative):
  comp_processed = process_text(narrative)
  prod_match_dict ={0:'Debt Collection', 1:'Credit Cards', 2:'Credit Reporting',3:'Retail Banking', 4:'Mortgages and Loans'}
  y = model.predict(TFIDF_Vect.transform([comp_processed]))
  return "This grieveance has to tagged to " + prod_match_dict.get(y[0]) + " department."

title = "Input the griveance to get it tagged"
description = """
<img src="https://gv-smdm.s3.amazonaws.com/logo.png" width=50> The model was trained to tag griveance based on consumerfinance.gov data.
To address the issue of mislabeled complaints, it’s important for companies to provide clear
and user-friendly complaint submission processes. This might include using plain language to
describe different types of complaints, providing examples or descriptions of common issues,
or even offering a guided submission process that helps customers identify the most appropriate
category for their concern.
"""

gr.Interface(
    fn=predict,
    inputs="textbox",
    outputs="text",
    title=title,
    description=description,
    examples=[["When I went to apply for this personal loan using car as collateral, it didn't advise me that the loan value had to be higher than the amount owed on the car loan, until after they ran my credit. There were no alerts or notifications about that potential before they ran my credit. I would like this hard inquiry removed because they were operating under a false premise and promised me a loan before giving me all the details. "], ["I have a Home Equity Line of Credit ( HELOC ) with Pen Fed Credit Union which was started on XX/XX/XXXX and which is up in XX/XX/XXXX. The account number is XXXX There was {$99000.00} of principal as of XX/XX/XXXX as per the Pen Fed web statement dated XX/XX/XXXX attached. This HELOC was never recorded. On XX/XX/XXXX I called Pen Fed at the Member Service Center Line XXXX to inquire what it would take to bring the account to XXXX on or before XX/XX/XXXX. I then mailed in check No. XXXX ( Copy attached ) dated XX/XX/XXXX for {$100000.00} which was more than the amount necessary to bring the account to XXXX. When that check was not debited from my account on XX/XX/XXXX, I called Pen Fed again to inquire what happened to the check. I spoke with XXXX and told him that 6 other checks in sequence were all dated XX/XX/XXXX and mailed at the same time in the same mailbox and that all were debited on or before XX/XX/XXXX. It seemed clear that Pen Fed was delaying the deposit to generate additional interest income. I told him that they had to deposit the check as I wanted the account closed. On XX/XX/XXXX, I printed the HELOC statement attached and called Pen Fed again and spoke with XXXX. I complained that they held the check without presenting it for payment. I demanded that the account be closed. She said that I did not follow instructions for a payoff and that they were going to return my check but deposited it after I called. I submit to you that Pen Fed has no right to delay presenting a check in payment of a loan and that they would have no right to return the check. I note that Pen Fed sent me by mail payoff instructions and insisted that it would have to be by wire or certified bank check. The actual date of clearance was immaterial to me because I paid this personally and expected that the account would be brought to zero and I could have it closed. The Pen Fed representative insisted that they would have to issue a satisfaction to be recorded. I told her that no satisfaction was required as the loan had never been recorded. To prove that I am correct I have attached a print out from the XXXX XXXX XXXX Official Records Search. That clearly shows no recording in favor of Pen Fed. That is when I was told that they would now record the loan and then file a satisfaction. I requested that a supervisor call me back but no one did. On XX/XX/XXXX received an email via XXXX secure messaging with a demand for a CASHIERS CHECK, CERTIFIED FUNDS OR WIRE [ transfer ] and demanding that I return the document with funds to close the line of credit at which time PENFED WILL PAYOFF THE LOAN AND INITIATE THE RELEASE OF LIEN ON THE PROPERTY. A copy of the message is attached. There is no provision to reply to this message so I called PenFed again and first spoke with XXXX who harangued me about the lien which would have to be released. She would not listen to me when I said there was no lien because the HELOC was never recorded. She then said that PenFed would record the loan ( even though it was a XXXX and I wanted it closed ) and then issue a satisfaction. When I told her I wanted the email address of an officer of PenFed to whom I could send a complaint she refused to provide any such email. She transferred me instead to supervisor XXXX who said the same thing. This call took more than 20 minutes and was another waste of my time. I told both representatives that I would be filing a complaint with the Consumer Financial Protection Bureau and they dont seem to care. I have advised the PenFed representatives that I have a new HELOC pending with another bank and that if PenFed now files a lien it will hold up my new HELOC and cause me significant damage. I asked the two representatives to give me their legal department and that request was refused. Account Activity section of my XXXX XXXX XXXX statement shows that check No. XXXX for {$100000.00} which was dated and mailed on XX/XX/XXXX was not debited until XX/XX/XXXX. Check No XXXX also dated and mailed on XX/XX/XXXX was debited on XX/XX/XXXX. Check Nos. XXXX, XXXX, XXXX, XXXX ( same account but different number series ) were all dated XX/XX/XXXX and debited between XXXX & XX/XX/XXXX. They were all mailed at the same time and in the same mail box as the check to Pen Fed. Check No. XXXX has an error on the date of XX/XX/XXXX because it was actually mailed on XX/XX/XXXX and debited on XX/XX/XXXX. The dates of the checks are confirmed by the Check Copies from the XX/XX/XXXX Statement. In addition, there were 3 check to XXXX dated and mailed on XX/XX/XXXX which were debited as ARC debits on XX/XX/XXXX. It is obvious that Pen Fed held check No. XXXX for {$100000.00} for several days to generate additional interest against my HELOC account. Would you kindly investigate PenFeds practices with regard to Home Equity Loans and their holding of checks to generate additional interest. The demand sent XX/XX/XXXX is dated XX/XX/XXXX and amounts to extortion. Unless you correct me with some statutory reference I believe that the unrecorded loan is no different than any other loan and that I am entitled to close the account after paying the full amount due. The submitted documents demonstrate that there is no balance due. The sequence of checks demonstrates that they held my check to generate additional interest and that, to me, is a fraud."]],
).launch()