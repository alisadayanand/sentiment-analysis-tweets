import re

#preprocesses each word using regex
def preprocessWord(word):
    #all punctuation is removed
    word = word.strip('\'"?!,.():;')
    #repitions of letters are restricted to being 2
    word = re.sub(r'(.)\1+', r'\1\1', word)
    #punctuation in words like - and ' removed
    word = re.sub(r'(-|\')', '', word)
    return word

#ensures word is valid ie. begins with a letter
def isWord(word):
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)

#replaces an emoji with its sentiment
def emoji(tweet):
    #positive emojis
    # :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    #negative emojis
    # :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet

#preprocesses tweets 
def preprocessTweet(tweet):
    processed_tweet = []
    #convert everything to lower case
    tweet = tweet.lower()
    #replaces website links with URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
    #replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', ' USER_MENTION ', tweet)
    #removes the hashtags
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    #remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    #replace ellipses or many fullstops with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    #remove spaces and quotation marks
    tweet = tweet.strip(' "\'')
    #replace emojis with either EMO_POS or EMO_NEG
    tweet = emoji(tweet)
    #replace multiple spaces with one space
    tweet = re.sub(r'\s+', ' ', tweet)
    words = tweet.split()
    
    #adds processed tweets to an array that can be added to a processed csv file
    for word in words:
        word = preprocessWord(word)
        if isWord(word):
            processed_tweet.append(word)

    return ' '.join(processed_tweet)

#adds processed tweets to a new CSV file
def preprocessCSV(csv, processed, test=False):
    save_to_file = open(processed, 'w')
    with open(csv, 'r') as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            tweet_id = line[:line.find(',')]
            #test and train csv files have different columns so need to be processed differently
            if not test:
                line = line[1 + line.find(','):]
                positive = int(line[:line.find(',')])
            line = line[1 + line.find(','):]
            tweet = line
            processed_tweet = preprocessTweet(tweet)
            if not test:
                save_to_file.write('%s,%d,%s\n' % (tweet_id, positive, processed_tweet))
            else:
                save_to_file.write('%s,%s\n' % (tweet_id, processed_tweet))

    save_to_file.close()
    print('\nProcessed tweets saved to: %s' % processed)
    return processed

if __name__ == '__main__':
    csv = 'train.csv'
    processed = 'train-processed.csv'
    preprocessCSV(csv, processed, test=False)
    csv = 'test.csv'
    processed = 'test-processed.csv'
    preprocessCSV(csv, processed, test=True)



