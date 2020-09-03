import random
import numpy as np
import pickle
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import lil_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from sklearn.metrics import plot_precision_recall_curve, plot_confusion_matrix
import matplotlib.pyplot as plt

#Naive Bayes Classifier

def extractFeatures(tweets, batch_size=500, test_file=True):
    num_batches = int(np.ceil(len(tweets) / float(batch_size)))

    for i in range(num_batches):
        batch = tweets[i * batch_size: (i + 1) * batch_size]
        features = lil_matrix((batch_size, 25000))
        labels = np.zeros(batch_size)

        for j, tweet in enumerate(batch):
            if test_file:
                tweet_words = tweet[1][0]
                tweet_bigrams = tweet[1][1]
            else:
                tweet_words = tweet[2][0]
                tweet_bigrams = tweet[2][1]
                labels[j] = tweet[1]

            for word in tweet_words:
                index = unigrams.get(word)
                if index:
                    features[j, index] += 1

            for bigram in tweet_bigrams:
                index = bigrams.get(bigram)
                if index:
                    features[j, 15000 + index] += 1
        yield features, labels


#Processes the CSV tweets file into  a list of tuples (tweet_id, feature_vector)
def processTweets(csv_file, test_file=True):
    tweets = []
    with open(csv_file, 'r') as csv:
        lines = csv.readlines()

        for i, line in enumerate(lines):
            if test_file:
                tweet_id, tweet = line.split(',')
            else:
                tweet_id, sentiment, tweet = line.split(',')

            #generate the feature vector
            uni_feature_vector = []
            bi_feature_vector = []
            words = tweet.split()
            for i in range(len(words) - 1):
                word = words[i]
                next_word = words[i + 1]
                if unigrams.get(word):
                    uni_feature_vector.append(word)

                if bigrams.get((word, next_word)):
                    bi_feature_vector.append((word, next_word))
            if len(words) >= 1:
                if unigrams.get(words[-1]):
                    uni_feature_vector.append(words[-1])
            feature_vector = uni_feature_vector, bi_feature_vector

            if test_file:
                tweets.append((tweet_id, feature_vector))
            else:
                tweets.append((tweet_id, int(sentiment), feature_vector))
    return tweets


if __name__ == '__main__':
    np.random.seed(30)

    #top unigrams in a dictionary of {word:rank} from the pickle files created
    with open("freqdist.pkl", 'rb') as pkl_file:
        freq_dist = pickle.load(pkl_file)
    most_common = freq_dist.most_common(15000)
    unigrams = {p[0]: i for i, p in enumerate(most_common)}


    #top bigrams in a dictionary of {bigram:rank} from the pickle files created
    with open("freqdist-bi.pkl", 'rb') as pkl_file:
        freq_dist = pickle.load(pkl_file)
    most_common = freq_dist.most_common(10000)
    bigrams = {p[0]: i for i, p in enumerate(most_common)}

    print('\nGenerating the feature vectors...')
    tweets = processTweets('train-processed.csv', test_file=False)
    #random 10% of tweets
    index = int((0.9) * len(tweets))
    random.shuffle(tweets)
    train_tweets, val_tweets= tweets[:index], tweets[index:]

    del tweets
    print('Extracting the features...')
    clf = MultinomialNB()
    batch_size = len(train_tweets)

    n_train_batches = int(np.ceil(len(train_tweets) / float(batch_size)))
    for training_set_X, training_set_y in extractFeatures(train_tweets, test_file=False, batch_size=batch_size):
        #tf-idf
        tfidf = TfidfTransformer(smooth_idf=True, sublinear_tf=True, use_idf=True).fit(training_set_X)
        training_set_X = tfidf.transform(training_set_X)
        clf.partial_fit(training_set_X, training_set_y, classes=[0, 1])

    test_tweets = processTweets('test-processed.csv', test_file=True)
    n_test_batches = int(np.ceil(len(test_tweets) / float(batch_size)))
    predictions = np.array([])
    for test_set_X, _ in extractFeatures(test_tweets, test_file=True):
        test_set_X = tfidf.transform(test_set_X)
        prediction = clf.predict(test_set_X)
        predictions = np.concatenate((predictions, prediction))
    predictions = [(str(j), int(predictions[j])) for j in range(len(test_tweets))]

    #new CSV created of type (tweet_id, positive)
    with open('naivebayes.csv', 'w') as csv:
        csv.write('id,prediction\n')
        for tweet_id, pred in predictions:
            csv.write(tweet_id)
            csv.write(',')
            csv.write(str(pred))
            csv.write('\n')
    print('Predictions saved to naivebayes.csv')

    #evaluations
    print('\nEvaluating model on 10% of tweets:')
    correct, total = 0, len(val_tweets)
    batch_size = len(val_tweets)
    n_val_batches = int(np.ceil(len(val_tweets) / float(batch_size)))

    for val_set_X, val_set_y in extractFeatures(val_tweets, test_file=False, batch_size=batch_size):
        val_set_X = tfidf.transform(val_set_X)
        prediction = clf.predict(val_set_X)

        correct += np.sum(prediction == val_set_y)
        f1 = f1_score(val_set_y, prediction, average="macro")
        precision = precision_score(val_set_y, prediction, average="macro")
        recall = recall_score(val_set_y, prediction, average="macro")
        average_precision = average_precision_score(val_set_y, prediction, average="macro")
        disp = plot_precision_recall_curve(clf, val_set_X, val_set_y)
        disp.ax_.set_title('Precision-Recall Curve')
        disp2 = plot_confusion_matrix(clf, val_set_X, val_set_y)
        disp2.ax_.set_title('Confusion Matrix')

    print('Accuracy: %d/%d = %.4f%%' % (correct, total, correct * 100. / total))
    print("Precision: %.4f" % precision)
    print("Recall: %.4f" % recall)
    print("F1 Score: %.4f" % f1)
    print("Precision-Recall Curve...")
    print("Confusion Matrix...")
    plt.show()

