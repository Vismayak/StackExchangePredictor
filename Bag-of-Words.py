import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
if __name__ == "__main__":

	data = pd.read_csv('Accepted_answer_prediction_data_train.txt', sep="\t", header=None)
	labels = pd.read_csv('Accepted_answer_prediction_labels_train.txt', sep="\t", header=None)
	data = data.iloc[:,1]	#onlt need text for bag of words I think
	labels = labels.iloc[:,1]

	BoW = set()

	for pos in range(int(0.85 * data.shape[0])):
		if labels[pos] == 1:
			for word in data[pos].split():
				BoW.add(word)

	print(BoW)
	pos = 0
	neg = 0
	for word in data[int(data.shape[0]) - 1].split():
		if word in BoW:
			pos += 1
		else:
			neg += 1

	print(pos)
	print(neg)
	print(len(data[data.shape[0] - 1].split()))
	print(labels[int(data.shape[0]) - 1])