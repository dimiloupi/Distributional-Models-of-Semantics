# coding: utf-8
import numpy 
from question1 import *
import json
from collections import Counter
'''
helper class to load a thesaurus from disk
input: thesaurusFile, file on disk containing a thesaurus of substitution words for targets
output: the thesaurus, as a mapping from target words to lists of substitution words
'''
def load_thesaurus(thesaurusFile):
	thesaurus = {}
	with open(thesaurusFile) as inFile:
		for line in inFile.readlines():
			word, subs = line.strip().split("\t")
			thesaurus[word] = subs.split(" ")
	return thesaurus

'''
(a) function addition for adding 2 vectors
input: vector1
input: vector2
output: addVector, the resulting vector when adding vector1 and vector2
'''
def addition(vector1, vector2):
	add_list = []
	if isinstance(vector1[0], tuple) and isinstance(vector2[0],tuple):#sparse vectors
		vec1 = Counter(dict(vector1))
		vec2 = Counter(dict(vector2))
		add = vec1 + vec2
		add_list = add.items()

	else:#full vectors
		for i, j in enumerate(vector1):
			for k,l in enumerate(vector2):
				x = (i, j+l)
				add_list.append(x)
	return add_list

'''
(a) function multiplication for multiplying 2 vectors
input: vector1
input: vector2
output: mulVector, the resulting vector when multiplying vector1 and vector2
'''
def multiplication(vector1, vector2):
	multi_list = []
	
        if isinstance(vector1[0], tuple) and isinstance(vector2[0],tuple):#sparse vectors
		vec1 = Counter(dict(vector1))
        	vec2 = Counter(dict(vector2))
		multi_d = {}
		for key, value in vec1.items():
			if key in vec2:
				multi_d[key] = vec1[key] * vec2[key]
				multi_list = multi_d.items()
	else:#full vectors
		for i,j in enumerate(vector1):
			for k,l in enumerate(vector2):
				if i==k:
					x = (i, j*l)
					multi_list.append(x)

	return multi_list
'''
(d) function prob_z_given_w to get probability of LDA topic z, given target word w
input: ldaModel
input: topicID as an integer
input: wordVector in frequency space
output: probability of the topic with topicID in the ldaModel, given the wordVector
'''
def prob_z_given_w(ldaModel, topicID, wordVector):
	# your code here
	return None

'''
(d) function prob_w_given_z to get probability of target word w, given LDA topic z
input: ldaModel
input: targetWord as a string
input: topicID as an integer
output: probability of the targetWord, given the topic with topicID in the ldaModel
'''
def prob_w_given_z(ldaModel, targetWord, topicID):
	# your code here
	return None

'''
(f) get the best substitution word in a given sentence, according to a given model (tf-idf, word2vec, LDA) and type (addition, multiplication, lda)
input: jsonSentence, a string in json format
input: thesaurus, mapping from target words to candidate substitution words
input: word2id, mapping from vocabulary words to word IDs
input: model, a vector space, Word2Vec or LDA model
input: frequency vectors, original frequency vectors (for querying LDA model)
input: csType, a string indicating the method of calculating context sensitive vectors: "addition", "multiplication", or "lda"
output: the best substitution word for the jsonSentence in the given model, using the given csType
'''
def best_substitute(jsonSentence, thesaurus, word2id, model, frequencyVectors, csType):

	# (b) use addition to get context sensitive vectors
	load_sentence = json.loads(jsonSentence)
	
	split_sentence = load_sentence["sentence"].split()

	if model == best_model2:#word2vec model
		word_target =list(model[load_sentence["target_word"]])#getting the vector for the target word
	if model == tfIdfVectors:
	
		word_target = list(model[word2id[load_sentence["target_word"]]])#getting the vector for the target word
		




	if csType == "addition":
		position_of_target = int(load_sentence["target_position"])
		positions = [-1, -2, -3, -4, -5, 1, 2, 3, 4, 5] #define the context window for the target word
		index = []
		top_score = 0
        	winner = ""
		subs = thesaurus[load_sentence["target_word"]]#get the substitutions words

		for i in positions:
			x = position_of_target + i
			index.append(x)#each item in the index list would be the position of the context word
			

		for sub in subs:
			score = 0

			for j in index:
				try:
					if model == best_model2:

						context_vector = model[split_sentence[j]]#get the vector for each context word

					else:
						
						context_vector = model[word2id[split_sentence[j]]]#get the vector for each context word


					add = addition(word_target, context_vector)
						
					if model == best_model2:
						w = sub
								
						score += cosine_similarity(model[w].tolist(),add)#sum up the cosine similarity for each substitution word
						#print score
							
					else:
						
						score += cosine_similarity(model[word2id[sub]], add)

					
						
				except:
					continue

				if score > top_score:

					top_score = score
                                        winner = sub
                                                

	
		return load_sentence["target_word"], load_sentence["id"], winner



	#(c) use multiplication to get context sensitive vectors
	elif csType == "multiplication":
		position_of_target = int(load_sentence["target_position"])
                positions = [-1, -2, -3, -4, -5, 1, 2, 3, 4, 5]#define the context window for the target word
                index = []
                top_score = 0
                subs = thesaurus[load_sentence["target_word"]]
                winner = ""

                for i in positions:
                        x = position_of_target + i
                        index.append(x)#each item in the index list would be the position of the context word


                for sub in subs:
                        score = 0

                        for j in index:
                                try:
                                        if model == best_model2:

                                                
                                                context_vector = model[split_sentence[j]]#get the vector for each context word

                                        else:

                                                context_vector = model[word2id[split_sentence[j]]]#get the vector for each context word


                                        multi = multiplication(word_target, context_vector)
					#print multi 					 
					
                                        if model == best_model2:
                                                w = sub
						
                                                score += cosine_similarity(model[w].tolist(), multi)#sum up the cosine similarity for each substitution word


                                        else:

                                                score += cosine_similarity(model[word2id[sub]], multi)



                                except:
                                        continue

                                if score > top_score:

                                        top_score = score
                                        winner = sub
                                        


        return load_sentence["target_word"], load_sentence["id"], winner


	# (d) use LDA to get context sensitive vectors
	#elif csType == "lda":
		# your code here
	#	pass


	return None

if __name__ == "__main__":
	import sys

	part = sys.argv[1]

	# this can give you an indication whether part a (vector addition and multiplication) works.
	if part == "a":
		print("(a): vector addition and multiplication")
		v1, v2, v3 , v4 = [(0,1), (2,1), (4,2)], [(0,1), (1,2), (4,1)], [1, 0, 1, 0, 2], [1, 2, 0, 0, 1]
		try:
			if not set(addition(v1, v2)) == set([(0, 2), (2, 1), (4, 3), (1, 2)]):
				print("\tError: sparse addition returned wrong result")
			else:
				print("\tPass: sparse addition")
		except Exception as e:
			print("\tError: exception raised in sparse addition")
			print(e)
		try:
			if not set(multiplication(v1, v2)) == set([(0,1), (4,2)]):
				print("\tError: sparse multiplication returned wrong result")
			else:
				print("\tPass: sparse multiplication")
		except Exception as e:
			print("\tError: exception raised in sparse multiplication")
			print(e)
		try:
			addition(v3,v4)
			print("\tPass: full addition")
		except Exception as e:
			print("\tError: exception raised in full addition")
			print(e)
		try:
			multiplication(v3,v4)
			print("\tPass: full multiplication")
		except Exception as e:
			print("\tError: exception raised in full addition")
			print(e)

	# you may complete this to get answers for part b (best substitution words with tf-idf and word2vec, using addition)
	if part == "b":
		print("(b) using addition to calculate best substitution words")
		id2word,word2id,vectors=load_corpus(sys.argv[2], sys.argv[3])
		tfIdfVectors = tf_idf(vectors)
		best_model2 = gensim.models.Word2Vec.load("/afs/inf.ed.ac.uk/user/s15/s1565257/virtualenvs/nlu/assignment1/bestmodel")
		thesauruS = load_thesaurus("/afs/inf.ed.ac.uk/user/s15/s1565257/virtualenvs/nlu/assignment1/data/test_thesaurus.txt")

		with open('word2vec_addition.txt', 'w') as wordaddition_file:
			with open("/afs/inf.ed.ac.uk/user/s15/s1565257/virtualenvs/nlu/assignment1/data/test.txt") as jason:
				for line in jason:
					x, y, z = best_substitute(line, thesauruS, word2id, best_model2, vectors, "addition")
					wordaddition_file.write("{} {} :: {}\n".format(x, y, z))
					

		with open('tf-idf_addition.txt', 'w') as tfaddition_file:
                        with open("/afs/inf.ed.ac.uk/user/s15/s1565257/virtualenvs/nlu/assignment1/data/test.txt") as jason:
                                for line in jason:
                                       x, y, z = best_substitute(line, thesauruS, word2id, tfIdfVectors, vectors, "addition")
                                       tfaddition_file.write("{} {} :: {}\n".format(x, y, z))

	# you may complete this to get answers for part c (best substitution words with tf-idf and word2vec, using multiplication)
	if part == "c":
		print("(c) using multiplication to calculate best substitution words")

		id2word,word2id,vectors=load_corpus(sys.argv[2], sys.argv[3])
                tfIdfVectors = tf_idf(vectors)
                best_model2 = gensim.models.Word2Vec.load("/afs/inf.ed.ac.uk/user/s15/s1565257/virtualenvs/nlu/assignment1/bestmodel")
                thesauruS = load_thesaurus("/afs/inf.ed.ac.uk/user/s15/s1565257/virtualenvs/nlu/assignment1/data/test_thesaurus.txt")

                with open('word2vec_multi.txt', 'w') as wordmulti_file:
                        with open("/afs/inf.ed.ac.uk/user/s15/s1565257/virtualenvs/nlu/assignment1/data/test.txt") as jason:
                                for line in jason:
                                        x, y, z = best_substitute(line, thesauruS, word2id, best_model2, vectors, "multiplication")
                                        wordmulti_file.write("{} {} :: {}\n".format(x, y, z))


                with open('tf-idf_multi.txt', 'w') as tfmulti_file:
                        with open("/afs/inf.ed.ac.uk/user/s15/s1565257/virtualenvs/nlu/assignment1/data/test.txt") as jason:
                                for line in jason:
                                       x, y, z = best_substitute(line, thesauruS, word2id, tfIdfVectors, vectors, "multiplication")
                                       tfmulti_file.write("{} {} :: {}\n".format(x, y, z))






	# this can give you an indication whether your part d1 (P(Z|w) and P(w|Z)) works
	if part == "d":
		print("(d): calculating P(Z|w) and P(w|Z)")
		print("\tloading corpus")
		id2word,word2id,vectors=load_corpus(sys.argv[2], sys.argv[3])
		print("\tloading LDA model")
		ldaModel = gensim.models.ldamodel.LdaModel.load("lda.model")
		houseTopic = ldaModel[vectors[word2id["house.n"]]][0][0]
		try:
			if prob_z_given_w(ldaModel, houseTopic, vectors[word2id["house.n"]]) > 0.0:
				print("\tPass: P(Z|w)")
			else:
				print("\tFail: P(Z|w)")
		except Exception as e:
			print("\tError: exception during P(Z|w)")
			print(e)
		try:
			if prob_w_given_z(ldaModel, "house.n", houseTopic) > 0.0:
				print("\tPass: P(w|Z)")
			else:
				print("\tFail: P(w|Z)")
		except Exception as e:
			print("\tError: exception during P(w|Z)")
			print(e)

	# you may complete this to get answers for part d2 (best substitution words with LDA)
	if part == "e":
		print("(e): using LDA to calculate best substitution words")
		# your code here
