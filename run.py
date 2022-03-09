from utils import *
import pprint
import math
import matplotlib.pyplot as plt
import random
def naive_bayes():

	###############################################################################################
	# QUESTION 1

	def product(pTrainPercent, nTrainPercent, pTestPercent, nTestPercent):
		percentage_positive_instances_train = pTrainPercent
		percentage_negative_instances_train = nTrainPercent

		percentage_positive_instances_test = pTestPercent
		percentage_negative_instances_test = nTestPercent

		(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train,
														  percentage_negative_instances_train)
		(pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

		# print("Number of positive training instances:", len(pos_train))
		# print("Number of negative training instances:", len(neg_train))
		# print("Number of positive test instances:", len(pos_test))
		# print("Number of negative test instances:", len(neg_test))

		## length of positive reviews vs length of negative reviews
		totalPosReviews = len(pos_train)
		totalNegReviews = len(neg_train)

		## yi for both the label
		probPos = totalPosReviews / (totalPosReviews + totalNegReviews)
		probNeg = totalNegReviews / (totalPosReviews + totalNegReviews)

		# dictionaries for frequency of words in both positive prob and negative prob
		posFreq = {}
		negFreq = {}

		posTotalwords = 0
		negTotalwords = 0

		uniqueWord = {}
		unique = 0
		vocabLen = len(vocab)

		with open('vocab.txt', 'w') as f:
			for word in vocab:
				f.write("%s\n" % word)

		print("Vocabulary (training set):", len(vocab))

		for i in pos_train:
			for j in i:
				posTotalwords += 1
				if j in posFreq:
					posFreq[j] += 1
				else:
					posFreq[j] = 1
			if j in uniqueWord:
				uniqueWord[j] += 1
			else:
				uniqueWord[j] = 1

		for i in neg_train:
			for j in i:
				negTotalwords += 1
				if j in negFreq:
					negFreq[j] += 1
				else:
					negFreq[j] = 1
			if j in uniqueWord:
				uniqueWord[j] += 1
			else:
				uniqueWord[j] = 1

		correctPositive = 0
		falsePositive = 0
		correctNegative = 0
		falseNegative = 0
		for i in pos_test:
			label1 = probPos
			label2 = probNeg
			for j in i:
				if j in posFreq:
					label1 *= (posFreq[j] / posTotalwords)
				if j in negFreq:
					label2 *= (negFreq[j] / negTotalwords)

			if label1 > label2:
				correctPositive += 1
			elif label2 > label1:
				falseNegative += 1
			else:
				if (random.randint(0,1) == 1):
					correctPositive+= 1
				else:
					falseNegative+= 1


		for i in neg_test:
			label1 = probPos
			label2 = probNeg
			for j in i:
				if j in posFreq:
					label1 *= (posFreq[j] / posTotalwords)
				if j in negFreq:
					label2 *= (negFreq[j] / negTotalwords)

			if label1 < label2:
				correctNegative += 1
			elif label2 > label1:
				falsePositive += 1
			else:
				if random.randint(0,1) == 1:
					correctNegative+= 1
				else:
					falsePositive+= 1
		return correctPositive, falsePositive, correctNegative, falseNegative

	##############################################################################################


	correctPositive, falsePositive, correctNegative, falseNegative = product(0.2,0.2,0.2,0.2)
	print(" \n \nQuestion 1. \n Using only the product we get :")
	n = correctNegative + correctPositive + falsePositive + falseNegative
	print("accuracy = ", (correctNegative + correctPositive) / (n))
	print("precision = ", correctPositive / (correctPositive + falsePositive))
	print("recall =", correctPositive / (correctPositive + falseNegative))
	print(
		f"Confusion Matrix = \nTP = {correctPositive} FN = {falseNegative} \nFP = {falsePositive} TN = {correctNegative}")

	##using Log##################################################################################################
	def log(alpha,pTrainPercent, nTrainPercent, pTestPercent, nTestPercent):
		percentage_positive_instances_train = pTrainPercent
		percentage_negative_instances_train = nTrainPercent

		percentage_positive_instances_test = pTestPercent
		percentage_negative_instances_test = nTestPercent

		(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train,
														  percentage_negative_instances_train)
		(pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

		# print("Number of positive training instances:", len(pos_train))
		# print("Number of negative training instances:", len(neg_train))
		# print("Number of positive test instances:", len(pos_test))
		# print("Number of negative test instances:", len(neg_test))

		## length of positive reviews vs length of negative reviews
		totalPosReviews = len(pos_train)
		totalNegReviews = len(neg_train)

		## yi for both the label
		probPos = totalPosReviews / (totalPosReviews + totalNegReviews)
		probNeg = totalNegReviews / (totalPosReviews + totalNegReviews)

		# dictionaries for frequency of words in both positive prob and negative prob
		posFreq = {}
		negFreq = {}

		posTotalwords = 0
		negTotalwords = 0

		uniqueWord = {}
		unique = 0
		vocabLen = len(vocab)

		for i in pos_train:
			for j in i:
				posTotalwords += 1
				if j in posFreq:
					posFreq[j] += 1
				else:
					posFreq[j] = 1
			if j in uniqueWord:
				uniqueWord[j] += 1
			else:
				uniqueWord[j] = 1

		for i in neg_train:
			for j in i:
				negTotalwords += 1
				if j in negFreq:
					negFreq[j] += 1
				else:
					negFreq[j] = 1
			if j in uniqueWord:
				uniqueWord[j] += 1
			else:
				uniqueWord[j] = 1

		correctPositive = 0
		falsePositive = 0
		correctNegative = 0
		falseNegative = 0
		for i in pos_test:
			label11 = math.log(probPos)
			label22 = math.log(probNeg)
			for j in i:
				if j in posFreq:
					label11 += math.log((posFreq[j] + alpha) / (posTotalwords + (alpha * vocabLen)))
				else:
					if alpha != 0:
						label11 += math.log(alpha / (posTotalwords + (alpha * vocabLen)))
				if j in negFreq:
					label22 += math.log((negFreq[j] + alpha) / (negTotalwords + (alpha * vocabLen)))
				else:
					if alpha != 0:
						label22 += math.log(alpha / (negTotalwords + (alpha * vocabLen)))

			if label11 > label22:
				correctPositive += 1
			elif label22 > label11:
				falseNegative += 1
			else:
				if (random.randint(0,1) == 1):
					correctPositive+= 1
				else:
					falseNegative+= 1

		for i in neg_test:
			label11 = math.log(probPos)
			label22 = math.log(probNeg)
			for j in i:
				if j in posFreq:
					label11 += math.log((posFreq[j] + alpha) / (posTotalwords + (alpha * vocabLen)))
				else:
					if alpha != 0:
						label11 += math.log( alpha / (posTotalwords + (alpha * vocabLen)))
				if j in negFreq:
					label22 += math.log((negFreq[j] + alpha) / (negTotalwords + (alpha * vocabLen)))
				else:
					if alpha!=0:
						label22 += math.log(alpha / (negTotalwords + (alpha * vocabLen)))


			if label11 < label22:
				correctNegative += 1
			elif label22 < label11:
				falsePositive += 1
			else:
				if (random.randint(0,1) == 1):
					correctNegative += 1
				else:
					falsePositive += 1

		return correctPositive, falsePositive, correctNegative, falseNegative

	correctPositive, falsePositive, correctNegative, falseNegative = log(0,pTrainPercent=0.2, nTrainPercent=0.2, pTestPercent=0.2, nTestPercent=0.2)
	print(" \n \nUsing only the log of product we get :")
	n = correctNegative + correctPositive + falsePositive + falseNegative
	print("accuracy = ", (correctNegative + correctPositive) / (n))
	print("precision = ", correctPositive / (correctPositive + falsePositive))
	print("recall =", correctPositive / (correctPositive + falseNegative))
	print(
		f"Confusion Matrix = \n TP = {correctPositive} FN = {falseNegative} \n FP = {falsePositive} TN = {correctNegative} \n \n \n")


	###########################################################################
	## QUESTION 2 #############################################################

	correctPositive, falsePositive, correctNegative, falseNegative = log(1,pTrainPercent=0.2, nTrainPercent=0.2, pTestPercent=0.2, nTestPercent=0.2)
	print(" \n \nQuestion 2. Using only the log of product with laplace smoothing and alpha = 1 we get :")
	n = correctNegative + correctPositive + falsePositive + falseNegative
	print("accuracy = ", (correctNegative + correctPositive) / (n))
	print("precision = ", correctPositive / (correctPositive + falsePositive))
	print("recall =", correctPositive / (correctPositive + falseNegative))
	print(
		f"Confusion Matrix = \n TP = {correctPositive} FN = {falseNegative} \n FP = {falsePositive} TN = {correctNegative} \n \n \n")


	## Part 2 ##################################################################

	def part2(pTrainPercent, nTrainPercent, pTestPercent, nTestPercent):
		percentage_positive_instances_train = pTrainPercent
		percentage_negative_instances_train = nTrainPercent

		percentage_positive_instances_test = pTestPercent
		percentage_negative_instances_test = nTestPercent

		(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train,
														  percentage_negative_instances_train)
		(pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

		# print("Number of positive training instances:", len(pos_train))
		# print("Number of negative training instances:", len(neg_train))
		# print("Number of positive test instances:", len(pos_test))
		# print("Number of negative test instances:", len(neg_test))

		## length of positive reviews vs length of negative reviews
		totalPosReviews = len(pos_train)
		totalNegReviews = len(neg_train)

		## yi for both the label
		probPos = totalPosReviews / (totalPosReviews + totalNegReviews)
		probNeg = totalNegReviews / (totalPosReviews + totalNegReviews)

		# dictionaries for frequency of words in both positive prob and negative prob
		posFreq = {}
		negFreq = {}

		posTotalwords = 0
		negTotalwords = 0

		uniqueWord = {}
		unique = 0
		vocabLen = len(vocab)

		for i in pos_train:
			for j in i:
				posTotalwords += 1
				if j in posFreq:
					posFreq[j] += 1
				else:
					posFreq[j] = 1
			if j in uniqueWord:
				uniqueWord[j] += 1
			else:
				uniqueWord[j] = 1

		for i in neg_train:
			for j in i:
				negTotalwords += 1
				if j in negFreq:
					negFreq[j] += 1
				else:
					negFreq[j] = 1
			if j in uniqueWord:
				uniqueWord[j] += 1
			else:
				uniqueWord[j] = 1

		def part2helper(alpha):
			correctPositive = 0
			falsePositive = 0
			correctNegative = 0
			falseNegative = 0
			for i in pos_test:
				label11 = math.log(probPos)
				label22 = math.log(probNeg)
				for j in i:
					if j in posFreq:
						label11 += math.log((posFreq[j] + alpha) / (posTotalwords + (alpha * vocabLen)))
					else:
						if alpha != 0:
							label11 += math.log(alpha / (posTotalwords + (alpha * vocabLen)))
					if j in negFreq:
						label22 += math.log((negFreq[j] + alpha) / (negTotalwords + (alpha * vocabLen)))
					else:
						if alpha != 0:
							label22 += math.log(alpha / (negTotalwords + (alpha * vocabLen)))

				if label11 > label22:
					correctPositive += 1
				elif label22 > label11:
					falseNegative += 1
				else:
					if (random.randint(0, 1) == 1):
						correctPositive += 1
					else:
						falseNegative += 1

			for i in neg_test:
				label11 = math.log(probPos)
				label22 = math.log(probNeg)
				for j in i:
					if j in posFreq:
						label11 += math.log((posFreq[j] + alpha) / (posTotalwords + (alpha * vocabLen)))
					else:
						if alpha != 0:
							label11 += math.log(alpha / (posTotalwords + (alpha * vocabLen)))
					if j in negFreq:
						label22 += math.log((negFreq[j] + alpha) / (negTotalwords + (alpha * vocabLen)))
					else:
						if alpha != 0:
							label22 += math.log(alpha / (negTotalwords + (alpha * vocabLen)))

				if label11 < label22:
					correctNegative += 1
				elif label22 < label11:
					falsePositive += 1
				else:
					if (random.randint(0, 1) == 1):
						correctNegative += 1
					else:
						falsePositive += 1
			return correctPositive, falsePositive, correctNegative, falseNegative

		alph = 0.0001
		alphas = []
		accuracies = []
		while (alph <= 1000):
			# alphas.append(math.log(alph))
			alphas.append(alph)
			correctPositive, falsePositive, correctNegative, falseNegative = part2helper(alph)
			n = correctNegative + correctPositive + falsePositive + falseNegative
			accuracy = (correctNegative + correctPositive) / (n)
			accuracies.append(accuracy)
			print(f"For alpha = {alph} accuracy = {accuracy}")
			alph *= 10
		return alphas,accuracies
	alphas,accuracies = part2(0.2,0.2,0.2,0.2)

	plt.title("Accuracies for different values of Alpha")
	plt.plot(alphas,accuracies)
	plt.xscale("log")
	plt.xlabel("Alpha values (log scale)")
	plt.ylabel("Accuracies")
	plt.show()
	maximizingAlpha = alphas[accuracies.index(max(accuracies))]

	##############################################################################
	# QUESTION 3
	print("\n \n \n Answer for question 3: ")

	correctPositive, falsePositive, correctNegative, falseNegative = log(maximizingAlpha,pTrainPercent=1, nTrainPercent=1, pTestPercent=1, nTestPercent=1)
	n = correctNegative + correctPositive + falsePositive + falseNegative
	print("\n \n accuracy = ", (correctNegative + correctPositive) / (n))
	print("precision = ", correctPositive / (correctPositive + falsePositive))
	print("recall =", correctPositive / (correctPositive + falseNegative))
	print(
		f"Confusion Matrix = \nTP = {correctPositive} FN = {falseNegative} \nFP = {falsePositive} TN = {correctNegative}")

	##############################################################################

	# Question 4
	print("\n \n \n \n Answer for question 4: ")
	correctPositive, falsePositive, correctNegative, falseNegative = log(10,pTrainPercent=0.5, nTrainPercent=0.5, pTestPercent=1, nTestPercent=1)
	n = correctNegative + correctPositive + falsePositive + falseNegative
	print("\n \n accuracy = ", (correctNegative + correctPositive) / (n))
	print("precision = ", correctPositive / (correctPositive + falsePositive))
	print("recall =", correctPositive / (correctPositive + falseNegative))
	print(
		f"Confusion Matrix = \nTP = {correctPositive} FN = {falseNegative} \nFP = {falsePositive} TN = {correctNegative}")

	##############################################################################

	#Question 6
	print("\n \n \n \n Answer for question 6: ")
	correctPositive, falsePositive, correctNegative, falseNegative = log(10, pTrainPercent=0.1, nTrainPercent=0.5,
																		 pTestPercent=1, nTestPercent=1)
	n = correctNegative + correctPositive + falsePositive + falseNegative
	print("\n \n accuracy = ", (correctNegative + correctPositive) / (n))
	print("precision = ", correctPositive / (correctPositive + falsePositive))
	print("recall =", correctPositive / (correctPositive + falseNegative))
	print(
		f"Confusion Matrix = \nTP = {correctPositive} FN = {falseNegative} \nFP = {falsePositive} TN = {correctNegative}")


if __name__=="__main__":
	naive_bayes()
