import csv
import pandas as pd
import mailbox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import re





#INITIALISE
#######################################################
df = pd.read_csv('data.csv') 
df = pd.DataFrame(df)
df = df.sample(n=10000)
col = ['label','url']
df = df[col]
#Deleting nulls
df = df[pd.notnull(df['url'])]
df.columns = ['label', 'url']
df['category_id'] = df['label'].factorize()[0]
category_id_df = df[['label', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'label']].values)

def get_tokens(inp):
    tokens_slash = str(inp.encode('utf-8')).split('/')
    all=[]
    for i in tokens_slash:
        tokens = str(i).split('-')
        tokens_dot = []
        for j in range(0,len(tokens)):
            temp = str(tokens[j]).split('.')
            tokens_dot = tokens_dot + temp
        all = all + tokens + tokens_dot
    all = list(set(all))
    if 'com' in all:
        all.remove('com')
    return all


vectorizer = TfidfVectorizer(tokenizer=get_tokens ,use_idf=True, smooth_idf=True, sublinear_tf=False)
features = vectorizer.fit_transform(df.url).toarray()

model = LogisticRegression(random_state=0)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, df.label, df.index, test_size=0.20, random_state=0)
model.fit(X_train, y_train)

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


	def extract_url(self):
		urls = re.findall(r"\b((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-]*)*/?)\b", self.get_body())
		return urls

	def predict_url(self):
		urls = self.extract_url()
		if urls == []:
			return ["good"]
		vect = vectorizer.transform(urls).toarray()
		return model.predict(vect)
			
	

class Mail:
	def __init__(self, mbox):
		self.mbox = mailbox.mbox(mbox)

	def predict_url(self):
		with open('url_result.csv', 'w', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(["Date", "To", "From", "Subject", "Content", "URLs"])
			for i in self.mbox:
				message = Message(i)
				result = message.predict_url()
				if 'bad' in result: #is spam
					writer.writerow([message.date, message.to, message.frm, message.subject, message.get_body(), "\n".join(message.extract_url())])
		return
		

		
