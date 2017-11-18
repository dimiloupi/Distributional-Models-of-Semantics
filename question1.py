# coding: utf-8

import gensim
import math
from copy import copy
from math import log,sqrt
from collections import defaultdict
'''
(f) helper class, do not modify.
provides an iterator over sentences in the provided BNC corpus
input: corpus path to the BNC corpus
input: n, number of sentences to retrieve (optional, standard -1: all)
'''
class BncSentences:
	def __init__(self, corpus, n=-1):
		self.corpus = corpus
		self.n = n

	def __iter__(self):
		n = self.n
		ret = []
		for line in open(self.corpus):
			line = line.strip().lower()
			if line.startswith("<s "):
				ret = []
			elif line.strip() == "</s>":
				if n > 0:
					n -= 1
				if n == 0:
					break
				yield copy(ret)
			else:
				parts = line.split("\t")
				if len(parts) == 3:
					word = parts[-1]
					idx = word.rfind("-")
					word, pos = word[:idx], word[idx+1:]
					if word in ['thus', 'late', 'often', 'only', 'usually', 'however', 'lately', 'absolutely', 'hardly', 'fairly', 'near', 'similarly', 'sooner', 'there', 'seriously', 'consequently', 'recently', 'across', 'softly', 'together', 'obviously', 'slightly', 'instantly', 'well', 'therefore', 'solely', 'intimately', 'correctly', 'roughly', 'truly', 'briefly', 'clearly', 'effectively', 'sometimes', 'everywhere', 'somewhat', 'behind', 'heavily', 'indeed', 'sufficiently', 'abruptly', 'narrowly', 'frequently', 'lightly', 'likewise', 'utterly', 'now', 'previously', 'barely', 'seemingly', 'along', 'equally', 'so', 'below', 'apart', 'rather', 'already', 'underneath', 'currently', 'here', 'quite', 'regularly', 'elsewhere', 'today', 'still', 'continuously', 'yet', 'virtually', 'of', 'exclusively', 'right', 'forward', 'properly', 'instead', 'this', 'immediately', 'nowadays', 'around', 'perfectly', 'reasonably', 'much', 'nevertheless', 'intently', 'forth', 'significantly', 'merely', 'repeatedly', 'soon', 'closely', 'shortly', 'accordingly', 'badly', 'formerly', 'alternatively', 'hard', 'hence', 'nearly', 'honestly', 'wholly', 'commonly', 'completely', 'perhaps', 'carefully', 'possibly', 'quietly', 'out', 'really', 'close', 'strongly', 'fiercely', 'strictly', 'jointly', 'earlier', 'round', 'as', 'definitely', 'purely', 'little', 'initially', 'ahead', 'occasionally', 'totally', 'severely', 'maybe', 'evidently', 'before', 'later', 'apparently', 'actually', 'onwards', 'almost', 'tightly', 'practically', 'extremely', 'just', 'accurately', 'entirely', 'faintly', 'away', 'since', 'genuinely', 'neatly', 'directly', 'potentially', 'presently', 'approximately', 'very', 'forwards', 'aside', 'that', 'hitherto', 'beforehand', 'fully', 'firmly', 'generally', 'altogether', 'gently', 'about', 'exceptionally', 'exactly', 'straight', 'on', 'off', 'ever', 'also', 'sharply', 'violently', 'undoubtedly', 'more', 'over', 'quickly', 'plainly', 'necessarily']:
						pos = "r"
					if pos == "j":
						pos = "a"
					ret.append(gensim.utils.any2unicode(word + "." + pos))

'''
(a) function load_corpus to read a corpus from disk
input: vocabFile containing vocabulary
input: contextFile containing word contexts
output: id2word mapping word IDs to words
output: word2id mapping words to word IDs
output: vectors for the corpus, as a list of sparse vectors
'''
def load_corpus(vocabFile="data/vocabulary.txt", contextFile="data/word_contexts.txt"):
	id2word = {}
	word2id = {}
	vectors = []

	with open(vocabFile, "r")	as f:


		for i, l  in enumerate(f):
			id2word[i] = l.strip()
			word2id[l.strip()] = i

	with open(contextFile, "r") as d:
		for count, line in enumerate(d):
			s_line = line.split()
			vector_list = []
			for count in range(1,len(s_line)):
				(index, freq) = s_line[count].split(":")
				vector_list.append((int(index),int(freq)))
			
			vectors.append(vector_list)




	return id2word, word2id, vectors

'''
(b) function cosine_similarity to calculate similarity between 2 vectors
input: vector1
input: vector2
output: cosine similarity between vector1 and vector2 as a real number
'''
def cosine_similarity(vector1, vector2):

	prodvw = 0
	v = 0
	w = 0
	if type(vector1[0]) == tuple and type(vector2[0]) == tuple:#for sparse vectors
       		 for (key,value) in vector1:
                	v += pow(value,2)

       		 for (key,value) in vector2:
               		 w += pow(value,2)

       		 for (key,value) in  vector1:
               		 for (key2,value2) in vector2:
                        		if(key==key2):
                               			 prodvw +=  value * value2


	else: #make the transformation
		if type(vector1[0]) == tuple:
			helpvector = []
			vector1 = dict(vector1)
			for item in range(len(vector2)):
				if item in vector1:
					helpvector.append(item)
				else:
					helpvector.append(0)
			vector1 = helpvector


		if type(vector2[0]) == tuple:
			helpvector = []
			vector2 = dict(vector2)
			for item in range(len(vector1)):
				if item in vector2:
					helpvector.append(item)
				else:
					helpvector.append(item)
			vector2 = []
			vector2 += helpvector

		for index, value in enumerate(vector1):
			for i, val in enumerate(vector2):
				if index == i:

					prodvw += value * val

		for value in vector1:
			v += pow(value,2)

		for value in vector2:
			w += pow(value,2)



	cos_sim =  prodvw / (sqrt(v)*sqrt(w))
	return cos_sim
'''
(d) function tf_idf to turn existing frequency-based vector model into tf-idf-based vector model
input: freqVectors, a list of frequency-based vectors
output: tfIdfVectors, a list of tf-idf-based vectors
'''
def tf_idf(freqVectors):
	tfIdfVectors = []
	N = float(20000)
	dft = []

	df0_dic = defaultdict(int)


	#compute the df values for every document/word
	for doc in freqVectors:#list
		for tup in doc:
			df0_dic[tup[0]] += 1


	#compute the tf value

	for doc in freqVectors:

		help_list = []
		for tup in doc:
			x = (tup[0], ((1 + log(tup[1], 2)) * (1 + log(N/float(df0_dic[tup[0]]), 2))))
			help_list.append(x)
		tfIdfVectors.append(help_list)
	
	
	return tfIdfVectors

'''
(f) function word2vec to build a word2vec vector model with 100 dimensions and window size 5
'''
def word2vec(corpus, learningRate, downsampleRate, negSampling):


	model1 = gensim.models.Word2Vec(corpus, size=100, alpha=learningRate, window=5, sample=downsampleRate, workers=5, negative=negSampling)



	return model1

'''
(h) function lda to build an LDA model with 100 topics from a frequency vector space
input: vectors
input: wordMapping mapping from word IDs to words
output: an LDA topic model with 100 topics, using the frequency vectors
'''
def lda(vectors, wordMapping):
	ldamodel = gensim.models.ldamodel.LdaModel(corpus = vectors, num_topics = 100, id2word = wordMapping, passes = 10, update_every = 0)
	return ldamodel

'''i
(j) function get_topic_words, to get words in a given LDA topic
inpuit: ldaModel, pre-trained Gensim LDA model
input: topicID, ID of the topic for which to get topic words
input: wordMapping, mapping from words to IDs (optional)
'''
def get_topic_words(ldaModel, topicID):
	help_list = ldaModel.show_topic(topicid = topicID, topn=100)
	top_list = [i[0] for i in help_list]
	return top_list

if __name__ == '__main__':
	import sys

	part = sys.argv[1].lower()

	# these are indices for house, home and time in the data. Don't change.
	house_noun = 80
	home_noun = 143
	time_noun = 12

	# this can give you an indication whether part a (loading a corpus) works.
	# not guaranteed that everything works.
	if part == "a":
		print("(a): load corpus")
		try:
			id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
			if not id2word:
				print("\tError: id2word is None or empty")
				exit()
			if not word2id:
				print("\tError: id2word is None or empty")
				exit()
			if not vectors:
				print("\tError: id2word is None or empty")
				exit()
			print("\tPass: load corpus from file")
		except Exception as e:
			print("\tError: could not load corpus from disk")
			print(e)

		try:
			if not id2word[house_noun] == "house.n" or not id2word[home_noun] == "home.n" or not id2word[time_noun] == "time.n":
				print("\tError: id2word fails to retrive correct words for ids")
			else:
				print("\tPass: id2word")
		except Exception:
			print("\tError: Exception in id2word")
			print(e)

		try:
			if not word2id["house.n"] == house_noun or not word2id["home.n"] == home_noun or not word2id["time.n"] == time_noun:
				print("\tError: word2id fails to retrive correct ids for words")
			else:
				print("\tPass: word2id")
		except Exception:
			print("\tError: Exception in word2id")
			print(e)

	# this can give you an indication whether part b (cosine similarity) works.
	# these are very simple dummy vectors, no guarantee it works for our actual vectors.
	if part == "b":
		import numpy
		print("(b): cosine similarity")
		try:
			cos = cosine_similarity([(0,1), (2,1), (4,2)], [(0,1), (1,2), (4,1)])
			if not numpy.isclose(0.5, cos):
				print("\tError: sparse expected similarity is 0.5, was {0}".format(cos))
			else:
				print("\tPass: sparse vector similarity")
		except Exception:
			print("\tError: failed for sparse vector")
		try:
			cos = cosine_similarity([1, 0, 1, 0, 2], [1, 2, 0, 0, 1])
			if not numpy.isclose(0.5, cos):
				print("\tError: full expected similarity is 0.5, was {0}".format(cos))
			else:
				print("\tPass: full vector similarity")
		except Exception:
			print("\tError: failed for full vector")

	# you may complete this part to get answers for part c (similarity in frequency space)
	if part == "c":
		print("(c) similarity of house, home and time in frequency space")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		house_home = cosine_similarity(vectors[word2id["house.n"]], vectors[word2id["home.n"]])
		print "The cosine similarity between house and home is", house_home
		time_house = cosine_similarity(vectors[word2id["time.n"]], vectors[word2id["house.n"]])
		print "The cosine similarity between time and house is", time_house
		time_home = cosine_similarity(vectors[word2id["time.n"]], vectors[word2id["home.n"]])
		print "The cosine similarity between time and home is", time_home

	# this gives you an indication whether your conversion into tf-idf space works.
	# this does not test for vector values in tf-idf space, hence can't tell you whether tf-idf has been implemented correctly
	if part == "d":
		print("(d) converting to tf-idf space")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])

		try:
			tfIdfSpace = tf_idf(vectors)
			print len(vectors), len(tfIdfSpace)
			if not len(vectors) == len(tfIdfSpace):
				print("\tError: tf-idf space does not correspond to original vector space")
			else:
				print("\tPass: converted to tf-idf space")
		except Exception as e:
			print("\tError: could not convert to tf-idf space")
			print(e)

	# you may complete this part to get answers for part e (similarity in tf-idf space)
	if part == "e":
		print("(e) similarity of house, home and time in tf-idf space")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		tfIdfVectors = tf_idf(vectors)
		house_home = cosine_similarity(tfIdfVectors[word2id["house.n"]], tfIdfVectors[word2id["home.n"]])
                print "The similarity between house and home in tf-idf space is", house_home
                time_house = cosine_similarity(tfIdfVectors[word2id["time.n"]], tfIdfVectors[word2id["house.n"]])
                print "The similarity between time and house in tf-idf space is", time_house
                time_home = cosine_similarity(tfIdfVectors[word2id["time.n"]], tfIdfVectors[word2id["home.n"]])
		print "The similarity between time and home in tf-idf space is", time_home

	# you may complete this part for the first part of f (estimating best learning rate, sample rate and negative samplings)
	if part == "f1":
		print("(f1) word2vec, estimating best learning rate, sample rate, negative sampling")





		sentences_sample = BncSentences("data/bnc.vert", n = 50000)
		learning_rate = [0.02, 0.04, 0.05]
		sample_rate = [0.1, 0.020 ,0.00001]
		negative_sampling = [2, 4, 5, 10]
		for lrate in learning_rate:
			for sample in sample_rate:
				for neg in negative_sampling:
					model2 = word2vec(sentences_sample, lrate, sample, neg)
					m2_accuracy = model2.accuracy("data/accuracy_test.txt")
        				incorrect = 0
                			correct = 0
					item = m2_accuracy[-1]

					for key, value in item.items():
						if key == "incorrect":
							incorrect += len(value)
						if key == "correct":
							correct += len(value)
					results_accuracy = (float(correct) / float(incorrect + correct))
					print "Accuracy level is:", results_accuracy,"learning rate is:", lrate, "sample rate is:", sample, "negative sampling is:", neg

	# you may complete this part for the second part of f (training and saving the actual word2vec model)
	if part == "f2":
		import logging
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
		print("(f2) word2vec, building full model with best parameters. May take a while.")

		s = BncSentences("data/bnc.vert")
		best_model = word2vec(s, 0.05, 0.02, 10)
		best_model.save("/afs/inf.ed.ac.uk/user/s15/s1565257/virtualenvs/nlu/assignment1/bestmodel")



	# you may complete this part to get answers for part g (similarity in your word2vec model)
	if part == "g":
		print("(g): word2vec based similarity")

		best_model2 = gensim.models.Word2Vec.load("/afs/inf.ed.ac.uk/user/s15/s1565257/virtualenvs/nlu/assignment1/bestmodel")
		house_home = cosine_similarity(best_model2["house.n"], best_model2["home.n"])
		print "word2vec similarity between home and house is", house_home
		house_time = cosine_similarity(best_model2["house.n"], best_model2["time.n"])
		print "word2vec similarity between time and house is", house_time
		home_time = cosine_similarity(best_model2["home.n"], best_model2["time.n"])
                print "word2vec similarity between time and home is", home_time


	# you may complete this for part h (training and saving the LDA model)
	if part == "h":
		import logging
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
		print("(h) LDA model")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		lda_model = lda(vectors, id2word)
		lda_model.save("/afs/inf.ed.ac.uk/user/s15/s1565257/virtualenvs/nlu/assignment1/lda_model")

	# you may complete this part to get answers for part i (similarity in your LDA model)
	if part == "i":
		print("(i): lda-based similarity")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		lda_model = gensim.models.LdaModel.load("/afs/inf.ed.ac.uk/user/s15/s1565257/virtualenvs/nlu/assignment1/lda_model")
		house_home = cosine_similarity(lda_model[vectors[word2id["house.n"]]], lda_model[vectors[word2id["home.n"]]])
                print "lda similarity between home and house is", house_home
                house_time = cosine_similarity(lda_model[vectors[word2id["house.n"]]], lda_model[vectors[word2id["time.n"]]])
                print "lda similarity between time and house is", house_time
                home_time = cosine_similarity(lda_model[vectors[word2id["home.n"]]], lda_model[vectors[word2id["time.n"]]])
                print "lda similarity between time and home is", home_time


	# you may complete this part to get answers for part j (topic words in your LDA model)
	if part == "j":
		print("(j) get topics from LDA model")
		lda_model = gensim.models.LdaModel.load("/afs/inf.ed.ac.uk/user/s15/s1565257/virtualenvs/nlu/assignment1/lda_model")
		top_25 = get_topic_words(lda_model, 25)
		print "TOP 100 WORDS FOR TOPIC 25", top_25
