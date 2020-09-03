import pickle
from nltk import FreqDist
from collections import Counter

if __name__ == '__main__':
    #analysis of processed tweets in CSV file
    num_tweets, num_pos_tweets, num_neg_tweets = 0, 0, 0
    num_emojis, num_pos_emojis, num_neg_emojis = 0, 0, 0
    num_user_mention = 0
    num_urls= 0
    num_words= 0
    num_bigrams= 0
    all_words = []
    all_bigrams = []

    with open('train-processed.csv', 'r') as csv:
        lines = csv.readlines()
        num_tweets = len(lines)

        for i, line in enumerate(lines):
            id, pos, tweet = line.strip().split(',')
            pos = int(pos)

            if pos:
                num_pos_tweets += 1
            else:
                num_neg_tweets += 1

            analysis = {}
            analysis['user_mention'] = tweet.count('USER_MENTION')
            analysis['urls'] = tweet.count('URL')
            analysis['emoji_positive'] = tweet.count('EMO_POS')
            analysis['emoji_negative'] = tweet.count('EMO_NEG')

            tweet = tweet.replace('USER_MENTION', '').replace('URL', '')
            words = tweet.split()
            analysis['words'] = len(words)

            bigrams = []
            num_words = len(words)
            for i in range(num_words - 1):
                bigrams.append((words[i], words[i + 1]))
            analysis['bigrams'] = len(bigrams)

            num_pos_emojis += analysis['emoji_positive']
            num_neg_emojis += analysis['emoji_negative']
            num_emojis = num_pos_emojis + num_neg_emojis
            num_user_mention += analysis['user_mention']
            num_urls += analysis['urls']
            all_words.extend(words)
            num_bigrams += analysis['bigrams']
            all_bigrams.extend(bigrams)

    print ('\nAnalysing Processed Tweets')
    print ('Tweets - Total: %d, Positive: %d, Negative: %d' % (num_tweets, num_pos_tweets, num_neg_tweets))
    print('Emojis - Total: %d, Positive: %d, Negative: %d' % (num_emojis, num_pos_emojis, num_neg_emojis))
    print ('User user_mention Total:',num_user_mention)
    print ('URLs Total:',num_urls)
    print ('Bigrams Total:',num_bigrams)

    # calculates the frequency distribution of unigrams and bigrams and writes them to pickle files
    print('\nFrequency Distributions Saved')

    #unigrams
    freq_dist = FreqDist(all_words)
    pkl_file_name = 'freqdist.pkl'
    with open(pkl_file_name, 'wb') as pkl_file:
        pickle.dump(freq_dist, pkl_file)
    print('Unigram -', pkl_file_name)

    #bigrams
    freq_dict = {}
    for bigram in all_bigrams:
        if freq_dict.get(bigram):
            freq_dict[bigram] += 1
        else:
            freq_dict[bigram] = 1
    bigram_freq_dist = Counter(freq_dict)
    bi_pkl_file_name = 'freqdist-bi.pkl'
    with open(bi_pkl_file_name, 'wb') as pkl_file:
        pickle.dump(bigram_freq_dist, pkl_file)
    print ('Bigram -', bi_pkl_file_name)


