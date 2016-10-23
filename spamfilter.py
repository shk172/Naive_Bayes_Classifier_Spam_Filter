#Starter code for spam filter assignment in EECS349 Machine Learning
#Author: Prem Seetharaman (replace your name here)

import sys
import numpy as np
import os
import shutil
import math
import random
import scipy.stats
import matplotlib.pyplot as plt

def parse(text_file):
	#This function parses the text_file passed into it into a set of words. Right now it just splits up the file by blank spaces, and returns the set of unique strings used in the file. 
	content = text_file.read()
	return np.unique(content.split())

def writedictionary(dictionary, dictionary_filename):
	#Don't edit this function. It writes the dictionary to an output file.
	output = open(dictionary_filename, 'w')
	header = 'word\tP[word|spam]\tP[word|ham]\n'
	output.write(header)
	for k in dictionary:
		line = '{0}\t{1}\t{2}\n'.format(k, str(dictionary[k]['spam']), str(dictionary[k]['ham']))
		output.write(line)
		

def makedictionary(spam_directory, ham_directory, dictionary_filename):
	#Making the dictionary. 
	spam = [f for f in os.listdir(spam_directory) if os.path.isfile(os.path.join(spam_directory, f))]
	ham = [f for f in os.listdir(ham_directory) if os.path.isfile(os.path.join(ham_directory, f))]
	spam_prior_probability = len(spam)/float((len(spam) + len(ham)))
	
	words = {}
 	wordCount = {}

	#These for loops walk through the files and construct the dictionary. 
    #The dictionary, words, is constructed so that words[word]['spam'] gives 
    #the probability of observing that word, given we have a spam document 
    #P(word|spam), and words[word]['ham'] gives the probability of observing 
    #that word, given a hamd document P(word|ham). 

    #Each word is initialized into the dictionary with the initial value of 0.
    #There are two dictionaries, one holding each word's probabilities of appearing in
    #spam or ham emails, and the other one the number of times the word appeared in either
    #classes.
    
	for s in spam:
		for word in parse(open(spam_directory + s)):
			if word not in words:
				words[word] = {'spam': 0, 'ham': 0}
				wordCount[word] = {'spamCount': 1, 'hamCount': 0}
			else:
 				wordCount[word]['spamCount'] += 1                
	for h in ham:
		for word in parse(open(ham_directory + h)):
			if word not in words:
				words[word] = {'spam': 0, 'ham': 0}
    				wordCount[word] = {'spamCount': 0, 'hamCount': 1}
			else:
				wordCount[word]['hamCount'] += 1    
    
    #The sorting function uses logarithms to calculate the probability. In order to
    #fix the log(0) error, integer 1 has been added to both the numerator and
    #denominator while calculating the probability, since there is a chance that
    #the words will appear in the future ('Black swan event')
	for word in words:
         if wordCount[word]['spamCount'] == 0:
             words[word]['spam'] = float(1 / (float(len(spam)) + 1))
         else:
             words[word]['spam'] = (float(wordCount[word]['spamCount'])+1) / (float(len(spam)) + 1)
        
         if wordCount[word]['hamCount'] == 0:
             words[word]['ham'] = float(1 / (float(len(ham)) + 1))
         else:
             words[word]['ham'] = (float(wordCount[word]['hamCount']) + 1) / (float(len(ham)) + 1)
	#Write it to a dictionary output file.
	writedictionary(words, dictionary_filename)
	
	return words, spam_prior_probability

def is_spam(content, dictionary, spam_prior_probability):
    #This code uses Naive Bayes Classifier; there are two pairs of lists that
    #uses the two different equations to calculate the probability. The rest of the
    #program uses the equation 7, the logarithmic one.
    probabilitySpam = 0
    probabilityHam = 0
    tempList = []
    spamCount = 1
    hamCount = 1
    spamLogList = []
    hamLogList = []
    for word in content:
        if word not in tempList:
            #
            tempList.append(word)
            if word not in dictionary:
                continue
            else:
                #For equation 6, multiplying all the probability values together
                #for each of the two classes
                spamCount *= dictionary[word]['spam']
                hamCount *= dictionary[word]['ham']
                
                #For equation 7, adding the log of the probability values in lists
                #to be summed.
                spamLogList.append(math.log(dictionary[word]['spam']))                
                hamLogList.append(math.log(dictionary[word]['ham']))

    #Equation 6

#    probabilitySpam = spamCount * spam_prior_probability
#    probabilityHam = hamCount * (1 - spam_prior_probability) 

    #Equation 7
    probabilitySpam = math.log(spam_prior_probability) + sum(spamLogList)
    probabilityHam = math.log(1-spam_prior_probability) + sum(hamLogList)
    if probabilitySpam > probabilityHam:
        return True
    else:
        return False


def spamfilterpriorprobability(content, dictionary, spam_prior_probability):
    #This is the sorter that uses only prior probability. Returns True or False
    #solely based on the prior probability, which means that all emails will be
    #categorize one or the other, not mixture of the two.
    if spam_prior_probability >= .5:
		return True
    else:
		return False


#The first spamsort function uses the Naive Bayes Classifier and sorts the mails into
#one pair of directories. Spamsort2 uses only prior probability, and sorts the emails
#into another pair of directories.
def spamsort(mail_directory, spam_directory, ham_directory, dictionary, spam_prior_probability):
	mail = [f for f in os.listdir(mail_directory) if os.path.isfile(os.path.join(mail_directory, f))]
	for m in mail:
		content = parse(open(mail_directory + m))	      
		spam = is_spam(content, dictionary, spam_prior_probability)
		if spam:
			shutil.copy(mail_directory + m, spam_directory)
		else:
			shutil.copy(mail_directory + m, ham_directory)

def spamsort2(mail_directory, spam_directory, ham_directory, dictionary, spam_prior_probability):
    	mail = [f for f in os.listdir(mail_directory) if os.path.isfile(os.path.join(mail_directory, f))]
	for m in mail:
		content = parse(open(mail_directory + m))	
		spam2 = spamfilterpriorprobability(content, dictionary, spam_prior_probability)
  		if spam2:
			shutil.copy(mail_directory + m, spam_directory)
		else:
			shutil.copy(mail_directory + m, ham_directory)
   
#The experiment takes in 100 emails, randomly chosen from the ham and spam training sets. It then calls the two
#types of filtering methods, and then counts how many errors were made for both. It returns the pair of values,
#the error measure for the two methods. After calculation, it moves all the emails in the testing set
#back to their origins so that the experiment can be repeated without having to manually delete them
#every time. The emails in the sorted directories are also deleted.
def experiment(mail_directory, spam_directory, ham_directory, dictionary, spam_prior_probability,test_spam_directory,test_spam_directory2,test_ham_directory,test_ham_directory2):
    spam = [f for f in os.listdir(spam_directory) if os.path.isfile(os.path.join(spam_directory, f))]
    ham = [f for f in os.listdir(ham_directory) if os.path.isfile(os.path.join(ham_directory, f))]
    allEmails = ham + spam
    errorCount = 0
    errorCount2 = 0   
    for i in range(0, 100):
        j = random.randrange(0, len(allEmails))
        noOverlap = False
        while noOverlap == False:
            if allEmails[j] not in os.listdir(mail_directory):
                if allEmails[j] in spam:
                    shutil.move(spam_directory + allEmails[j], mail_directory)
                elif allEmails[j] in ham:
                    shutil.move(ham_directory + allEmails[j], mail_directory)
                noOverlap = True
            else:
                noOverlap = True
        
    spamsort(mail_directory, test_spam_directory, test_ham_directory, dictionary, spam_prior_probability)         
    for a in os.listdir(mail_directory):
        if a in ham and a in os.listdir(test_spam_directory):
            errorCount += 1
        elif a in spam and a in os.listdir(test_ham_directory):
            errorCount += 1
    errorSpamFilter = float(errorCount) / float(len(os.listdir(mail_directory)))
    
    spamsort2(mail_directory, test_spam_directory2, test_ham_directory2, dictionary, spam_prior_probability)         
    for b in os.listdir(mail_directory):
        if b in ham and b in os.listdir(test_spam_directory2):
            errorCount2 += 1
        elif b in spam and b in os.listdir(test_ham_directory2):
            errorCount2 += 1
    error = float(errorCount2) / float(len(os.listdir(mail_directory)))
    
    for c in os.listdir(mail_directory):
        if c in ham:
            shutil.move(mail_directory + c, ham_directory)
        elif c in spam:
            shutil.move(mail_directory + c, spam_directory)
    for d in os.listdir(test_spam_directory):
        os.remove(test_spam_directory + '/' + d)
    for e in os.listdir(test_ham_directory):
        os.remove(test_ham_directory + '/' + e)
    for f in os.listdir(test_ham_directory2):
        os.remove(test_ham_directory2 + '/' + f)
    for g in os.listdir(test_spam_directory2):
        os.remove(test_spam_directory2 + '/' + g)
    return errorSpamFilter, error    
    
    
if __name__ == "__main__":
	#Here you can test your functions. Pass it a training_spam_directory, 
    #a training_ham_directory, and a mail_directory that is filled with 
    #unsorted mail on the command line. It will create two directories in the 
    #directory where this file exists: sorted_spam, and sorted_ham. The files 
    #will show up  in this directories according to the algorithm you developed.
	training_spam_directory = sys.argv[1]
	training_ham_directory = sys.argv[2]	
	test_mail_directory = sys.argv[3]
	test_spam_directory = 'sorted_spam'
	test_ham_directory = 'sorted_ham'
 #Additional pair of directories for sorting using prior probability
	test_spam_directory2 = 'sorted_spam2'
	test_ham_directory2 = 'sorted_ham2'
	
	if not os.path.exists(test_spam_directory):
		os.mkdir(test_spam_directory)
	if not os.path.exists(test_ham_directory):
		os.mkdir(test_ham_directory)
  
	if not os.path.exists(test_spam_directory2):
		os.mkdir(test_spam_directory2)
	if not os.path.exists(test_ham_directory2):
		os.mkdir(test_ham_directory2)
	
	dictionary_filename = "dictionary.dict"
	
	#create the dictionary to be used
	dictionary, spam_prior_probability = makedictionary(training_spam_directory, training_ham_directory, dictionary_filename)
	tTestList1 = []
	tTestList2 = []
 #This repeats the classification for both Naive Bayes classifier and
 #Prior Probability classifier 100 times. Each iteration will sort 100 emails
 #and calculate how many times the two methods misclassified them.
	for i in range(0,100):
		errors = experiment(test_mail_directory, training_spam_directory, training_ham_directory, dictionary, spam_prior_probability, test_spam_directory, test_spam_directory2,test_ham_directory,test_ham_directory2)
		tTestList1.append(errors[0])
		tTestList2.append(errors[1])

#T-Test using the results from the experiment, calculates the p-value to see
#whether the two methods have statistically significant difference. The average
#error is calculated by dividing the sum of the errors by 100, the number of samples.
	pValue = scipy.stats.ttest_rel(tTestList1, tTestList2)
	averageError1 = sum(tTestList1) / 100
	averageError2 = sum(tTestList2) / 100
	print 'pValue: ', pValue[1]
	print 'Average Error for Spam Filter with Naive Bayes Classifier is: ', averageError1
	print 'Average Error for Spam Filter with Prior Probability is: ', averageError2
 
#Plots the errors of Naive Bayes Classifier against Prior Probability Method
	nbc, = plt.plot(range(100),tTestList1, '-r')
	pp, = plt.plot(range(100),tTestList2, '-b')
	nbc.set_label('Naive Bayes Classifier')
	pp.set_label('Prior Probability Filter')
	plt.xlabel('Samples')
	plt.ylabel('Average Error')
	plt.title('Average Error of for Each Samples')
	plt.legend()
	plt.show()
	#print pValue, 'pValue'
    #sort the mail
	#spamsort(test_mail_directory, test_spam_directory, test_ham_directory, dictionary, spam_prior_probability) 
