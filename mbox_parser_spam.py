import csv
import pandas as pd
import mailbox
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

#INITIALISE 
#######################################################

df = pd.read_csv("emails.csv")
messages = df["text"].tolist()
spam = df["spam"].values
processed_messages = []

#cleaning data 
#removes special character, digits and empty spaces
for message in messages:
    message = re.sub(r"\W", " ", message)
    message = re.sub(r"\s+[a-zA-Z]\s+", " ", message)
    message = re.sub(r"\^[a-zA-Z]\s+", " ", message)
    message = re.sub(r"\s+", " ", message, flags=re.I)
    message = re.sub(r"^b\s+", "", message)
    processed_messages.append(message)

X_train, X_test, y_train, y_test = train_test_split(processed_messages, spam, test_size=0.2, random_state=0)


vectorizer = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.75, stop_words=stopwords.words('english'))
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

spam_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
spam_classifier.fit(X_train, y_train)

##########################################################

class Message:
	def __init__(self, message):
		self.message = message
		self.frm = message['from']
		self.to = message['to']
		self.subject = message['subject']
		self.date = message["date"]
		#self.return_path = message["return-path"]

	def get_body(self): #getting plain text 'email body'
		body = None
		if self.message.is_multipart():
			for part in self.message.walk():
				if part.is_multipart():
					for subpart in part.walk():
						if subpart.get_content_type() == 'text/plain':
							body = subpart.get_payload(decode=True)
				elif part.get_content_type() == 'text/plain':
					body = part.get_payload(decode=True)
		elif self.message.get_content_type() == 'text/plain':
			body = self.message.get_payload(decode=True)
		return body.decode("utf-8") 

	def preprocess_message(self):
		message = "Subject: " + self.subject + " " + self.get_body()
		message = re.sub(r'\W', ' ', message)
		message = re.sub(r'\s+[a-zA-Z]\s+', ' ', message)
		message = re.sub(r'\^[a-zA-Z]\s+', ' ', message)
		message = re.sub(r'\s+', ' ', message, flags=re.I)
		message = re.sub(r'^b\s+', '', message)
		message = word_tokenize(message)
		message = [word for word in message if not word in stopwords.words('english')]
		return [" ".join(message)]
		
	def predict_spam(self):
		data = self.preprocess_message()
		vect = vectorizer.transform(data).toarray()
		return spam_classifier.predict(vect)

class Mail:
	def __init__(self, mbox):
		self.mbox = mailbox.mbox(mbox)



	def predict_spam(self):
		with open('spam_result.csv', 'w', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(["Date", "To", "From", "Subject", "Content"])
			for i in self.mbox:
				message = Message(i)
				result = message.predict_spam()
				if result == 1: #is spam
					writer.writerow([message.date, message.to, message.frm, message.subject, message.get_body()])



		
