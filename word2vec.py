import numpy as np
from utils import *


# Function to get word2vec representations
# Arguments:
# docs: A list of strings, each string represents a document
# Returns: mat (numpy.ndarray) of size (len(docs), dim)
# mat is a two-dimensional numpy array containing vector representation for ith document (in input list docs) in ith row
# dim represents the dimensions of word vectors, here dim = 300 for Google News pre-trained vectors
def word2vec_rep(docs):
	token_list = []
	stopwords = list(get_stopwords())
	for i in docs:
		i = i.lower()
		tokens = get_tokens(i)
		for token in tokens:
			if token not in stopwords:
				token_list.append(token)
	result_list =list(set(token_list))
	result_list.sort()
	matrix = np.zeros((len(docs), len(result_list)))
	for i in range(len(docs)):
		token = get_tokens(docs[i].lower())
		for j in range(len(result_list)):
			matrix[i][j] = token.count(result_list[j])
	# Dummy matrix
	dim = 300
	word_vectors = np.zeros((len(docs), dim))
	w2v = load_w2v()

	for i in range(len(docs)):
		divisor = 0
		for j in range(len(result_list)):
			if matrix[i][j] > 0:
				try:
					k=0
					while k < int(matrix[i][j]):
						word_vectors[i] = np.add(word_vectors[i],w2v[result_list[j]])
						divisor+=1
						k+=1
				except:
					continue
		try:
			if divisor > 0:
				word_vectors[i] = np.divide(word_vectors[i],divisor)
		except:
			continue
	# print(word_vectors)
	return word_vectors



def main():
	# Initialize the corpus
	sample_corpus = ['Many buildings at UIC are designed in the brutalist style.',
					'Brutalist buildings are generally characterized by stark geometric lines and exposed concrete.',
					'One famous proponent of brutalism was a Chicago architect named Walter Netsch.',
					'Walter Netsch designed the entire UIC campus in the early 1960s.',
					'When strolling the campus and admiring the brutalism, remember to think of Walter Netsch!']
	word2vec_rep(sample_corpus)
	# We can tokenize the first document as
	tokens = get_tokens(sample_corpus[0])
	# print("Tokens for first document: {0}".format(tokens))


	# We can fetch stopwords and check if a word is a stopword
	stopwords = get_stopwords()
	for word in ['he', 'hello', 'she', 'uic']:
		answer = word in stopwords
		# print("Is '{0}' a stopword? {1}".format(word, answer))

	# We can load numpy word vectors using load_w2v as
	# w2v = load_w2v()
	# And access these vectors using the dictionary
	# print(w2v['chicago'])



################ Do not make any changes below this line ################
if __name__ == '__main__':
	main()
