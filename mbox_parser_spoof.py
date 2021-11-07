import csv
import mailbox

class Message:
	def __init__(self, message):
		self.message = message
		self.frm = message['from']
		self.to = message['to']
		self.subject = message['subject']
		self.date = message["date"]
		try:
			self.return_path = message["return-path"]
		except:
			self.return_path = None
		try:
			self.reply_to = message["reply-to"]
		except:
			self.reply_to = None

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


	def predict_spoof(self):
		if self.return_path == None or self.to == None:
			return False
		else:
			if self.return_path != self.to:
				return True

			else:
				return False
			
	

class Mail:
	def __init__(self, mbox):
		self.mbox = mailbox.mbox(mbox)

	def predict_url(self):
		with open('spoof_result.csv', 'w', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(["Date", "To", "From", "Subject", "Content", "Reply-To", "Return-Path"])
			for i in self.mbox:
				message = Message(i)
				result = message.predict_spoof()
				if result: #might be spoofed
					writer.writerow([message.date, message.to, message.frm, message.subject, message.get_body(), self.reply_to, self.return_path])
		return
		



		