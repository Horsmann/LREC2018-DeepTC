from sys import argv
import numpy as np
import sys
import argparse

np.set_printoptions(threshold=np.nan)

def numpyizeVector(vec):
	vout=[]
	file = open(vec, 'r')
	for l in file.readlines():
		l = l.strip()
		v = [int(x) for x in l.split()]
		vout.append(v)
	file.close()
	return np.asarray(vout)
	
def loadEmbeddings(emb):
	matrix = {}	
	f = open(emb, 'r')
	embData = f.readlines()
	f.close()
	dim = len(embData[0].split())-1
	matrix = np.zeros((len(embData)+1, dim))	
	for e in embData:
		e = e.strip()
		if not e:
			continue
		idx = e.find(" ")
		id = e[:idx]
		vector = e[idx+1:]
		matrix[int(id)]=np.asarray(vector.split(" "), dtype='float32')
	return matrix, dim
	
def getMaxLabel(trainLabel, testLabel):
	maxLabel=-1
	for e in [trainLabel, testLabel]:
		for s in e:
			for v in s:
				if v>maxLabel:
					maxLabel=v	
	return maxLabel+1
	
def getMaxVocab(trainData, testData):
	vocabSize=-1
	for e in [trainData, testData]:
		for s in e:
			for v in s:
				if v>vocabSize:
					vocabSize=v	
	return vocabSize

def runExperiment(seed, trainVec, trainOutcome, testVec, testOutcome, embedding, longest_sequence, predictionOut):	

	np.random.seed(seed)

	from keras.preprocessing import sequence
	from keras.models import Sequential
	from keras.layers import Dense, Activation, Embedding, TimeDistributed, Bidirectional, Dropout
	from keras.layers import LSTM
	from keras.utils import np_utils
	from keras.optimizers import SGD
	

	trainVecNump = numpyizeVector(trainVec)
	trainOutcome = numpyizeVector(trainOutcome)
	
	testVecNump = numpyizeVector(testVec)
	testOutcome = numpyizeVector(testOutcome)
    
	embeddings,dim = loadEmbeddings(embedding)
	EMBEDDING_DIM = dim
	
	x_train=trainVecNump
	y_train=trainOutcome
	
	x_test = testVecNump
	y_test = testOutcome
	
	maxLabel=getMaxLabel(trainOutcome, testOutcome)
		
	vocabSize=getMaxVocab(trainVecNump, testVecNump)
				
	# training label to one-hot vectors
	y_train = np.array([np_utils.to_categorical(s, maxLabel) for s in y_train])

	print("Building model")
	model = Sequential()
	model.add(Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0],
                        weights=[embeddings],
                        trainable=False))                        
	model.add(Bidirectional(LSTM(100, return_sequences=True, activation="tanh",kernel_initializer="glorot_uniform")))
	model.add(TimeDistributed(Dense(maxLabel)))
	model.add(Activation('softmax'))
	sgdOptimizer=SGD(lr=0.1, momentum=0., decay=0., nesterov=False)

	# try using different optimizers and different optimizer configs
	model.compile(loss='categorical_crossentropy',
              optimizer=sgdOptimizer,
              metrics=['accuracy'])

	
	print("Start training")
	for i in range(0, 20):
		randomize = np.arange(len(x_train))
		np.random.shuffle(randomize)
		x_train = x_train[randomize]
		y_train = y_train[randomize]
		assert(len(x_train) == len(y_train))
		for c, (x,y) in enumerate(zip(x_train, y_train)):
			x=np.asarray([x])
			y=np.asarray([y])
			
			#sys.stdout.write('\r')
			#sys.stdout.write("%.1f %% of data provided" % ((c/(len(x_train)-1))*100))
			#sys.stdout.flush()
			
			a = model.fit(x, y, batch_size=1, verbose=0)
		print("\nEpoche " + str(i+1) + " completed")
		
	prediction = [model.predict_classes(np.asarray([x]), verbose=0) for x in x_test]
	
	predictionFile = open(predictionOut, 'w')
	predictionFile.write("#Gold\tPrediction\n")
	for i in range(0, len(prediction)):
		predictionEntry = prediction[i][0]
		for j in range(0, len(y_test[i])):
			if y_test[i][j]==0:
				continue #we reached the padded area - zero is reserved
			predictionFile.write(str(y_test[i][j]) +"\t" + str(predictionEntry[j]))
			if j+1 < len(y_test[i]):
				predictionFile.write("\n")
		predictionFile.write("\n")
	predictionFile.close()


if  __name__ =='__main__':
	parser = argparse.ArgumentParser(description="LREC Keras")
	parser.add_argument("--trainData", nargs=1, required=True)
	parser.add_argument("--trainOutcome", nargs=1, required=True)
	parser.add_argument("--testData", nargs=1, required=True)
	parser.add_argument("--testOutcome", nargs=1, required=True)    
	parser.add_argument("--embedding", nargs=1, required=True)    
	parser.add_argument("--maxLen", nargs=1, required=True)
	parser.add_argument("--predictionOut", nargs=1, required=True)
	parser.add_argument("--seed", nargs=1, required=True)    
    
    
	args = parser.parse_args()
    
	trainData = args.trainData[0]
	trainOutcome = args.trainOutcome[0]
	testData = args.testData[0]
	testOutcome = args.testOutcome[0]
	embedding = args.embedding[0]
	maxLen = args.maxLen[0]
	predictionOut = args.predictionOut[0]
	seed = args.seed[0]
	
	runExperiment(int(seed), trainData, trainOutcome, testData, testOutcome, embedding, int(maxLen), predictionOut)
