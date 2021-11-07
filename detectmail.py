import argparse

def spam(file): #takes in mbox file and returns csv
	import mbox_parser_spam as mp
	m = mp.Mail(file)
	m.predict_spam()
	return

def url(file):
	import mbox_parser_url as mp
	m = mp.Mail(file)
	m.predict_url()
	return

def spoof(file):
	import mbox_parser_spoof as mp
	m = mp.Mail(file)
	m.predict_spoof()
	return


def main():
	print("Processing mbox file...")
	parser = argparse.ArgumentParser(description = "Tool to detect suspicious emails.\nUsage: detectmail <option> <mbox_file>\nOptions: spam, url, spoof")

	parser.add_argument('command',
	type=str,
	help="Enter option: ")

	parser.add_argument('mbox_path', type=str, help="Enter mobx file path")
	args = parser.parse_args()

	command = args.command
	file = args.mbox_path


	if command == "spam":
		spam(file)
		print("Result saved as spam_result.csv")
		return

	elif command == "url":
		url(file)
		print("Result saved as url_result.csv")
		return
	elif command == "spoof":
		spoof(file)
		print("Result saved as spoof_result.csv")
		return
	else:
		print("Usage: detectmail <option> <mbox_file>")
		return




main()


