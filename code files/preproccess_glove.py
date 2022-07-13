import pandas as pd
import numpy as np

from collections import Counter
import pickle
import re



def create_weights(target_vocab, glove):
    matrix_len = len(target_vocab) + 1
    weights_matrix = np.zeros((matrix_len, 100))
    words_found = 0
    for i, word in enumerate(target_vocab):
        try:
            weights_matrix[i + 1] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(100,))
    return weights_matrix


def preprocess_glove_embs():
    train = pd.read_csv('TrainEmotions.csv')
    test = pd.read_csv('TestEmotions.csv')

    np_vectors = np.zeros(((1193514, 100)))
    words = []
    idx = 0
    word2idx = {}
    vectors = []

    with open(f'glove.twitter.27B.100d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            vect = np.array(line[1:]).astype(np.float)
            if len(vect) != 100:
                continue
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vectors.append(vect)


    vectors = np.array(vectors, dtype=object)
    pickle.dump(vectors, open(f'vectors.pkl', 'wb'))
    pickle.dump(words, open(f'words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'word2idx.pkl', 'wb'))

    glove = {w: vectors[word2idx[w]] for w in words}

    glove_df = pd.DataFrame.from_dict(glove, orient='index')

    glove_df.to_csv('./glove.csv', index=True, header=False)


def tokenize_line(line, vocab_to_int):
    return [vocab_to_int[w] for w in line.split()]


def pad_features(tweet, seq_length=161):
    ''' Return features matrix 2D of tweets_int, where each tweet is padded with 0's or truncated
    to the input seq_length.'''
    tweet_len = len(tweet)
    if tweet_len <= seq_length:
        zeroes = list(np.zeros(seq_length - tweet_len))
        new = zeroes + tweet ## zeroes in the begining
    else:
        new = tweet[0:seq_length]
    return np.array(new)


def create_vocabulary(all_text):
    all_text2 = ' '.join(all_text)
    # create a list of words
    words = all_text2.split()
    # Count all the words using Counter Method
    count_words = Counter(words)
    total_words = len(words)
    sorted_words = count_words.most_common(total_words)

    ##  later on we are going to do padding for shorter reviews and conventional choice for padding is 0.
    # So we need to start this indexing from 1
    vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
    int_to_vocab = {str(value) : key for key, value in vocab_to_int.items()}
    return vocab_to_int, int_to_vocab


def sub(pattern, output, string, whole_word=False):
    token = output
    if whole_word:
        pattern = r'(\s|^)' + pattern + r'(\s|$)'

    if isinstance(output, str):
        token = ' ' + output + ' '
    else:
        token = lambda match: ' ' + output(match) + ' '
    return re.sub(pattern, token, string)


def hashtag(token):
    token = token.group('tag')
    if token != token.upper():
        token = ' '.join(re.findall('[a-zA-Z][^A-Z]*', token))
    return '<hashtag> ' + token + ' <endhashtag>'


def punc_repeat(token):
    return token.group(0)[0] + " <repeat>"


def punc_separate(token):
    return token.group()


def number(token):
    return ' <number>'


def word_end_repeat(token):
    return token.group(1) + token.group(2) + ' <elong>'


def preprocess_glove_line(tweet):
    eyes = r"[8:=;]"
    nose = r"['`\-\^]?"
    sad_front = r"[(\[/\\]+"
    sad_back = r"[)\]/\\]+"
    smile_front = r"[)\]]+"
    smile_back = r"[(\[]+"
    lol_front = r"[DbpP]+"
    lol_back = r"[d]+"
    neutral = r"[|]+"
    sadface = eyes + nose + sad_front + '|' + sad_back + nose + eyes
    smile = eyes + nose + smile_front + '|' + smile_back + nose + eyes
    lolface = eyes + nose + lol_front + '|' + lol_back + nose + eyes
    neutralface = eyes + nose + neutral + '|' + neutral + nose + eyes
    punctuation = r"""[ '!"#$%&'()+,/:;=?@_`{|}~\*\-\.\^\\\[\]]+"""  ## < and > omitted to avoid messing up tokens

    tweet = sub(r'[\s]+', '  ', tweet)  # ensure 2 spaces between everything
    tweet = sub(r'(?:(?:https?|ftp)://|www\.)[^\s]+', '<url>', tweet, True)
    tweet = sub(r'@\w+', '<user>', tweet, True)
    tweet = sub(r'#(?P<tag>\w+)', hashtag, tweet, True)
    tweet = sub(sadface, '<sadface>', tweet, True)
    tweet = sub(smile, '<smile>', tweet, True)
    tweet = sub(lolface, '<lolface>', tweet, True)
    tweet = sub(neutralface, '<neutralface>', tweet, True)
    tweet = sub(r'(?:<3+)+', '<heart>', tweet, True)
    tweet = tweet.lower()
    tweet = sub(r'[-+]?[.\d]*[\d]+[:,.\d]*', number, tweet, True)
    tweet = sub(punctuation, punc_separate, tweet)
    tweet = sub(r'([!?.])\1+', punc_repeat, tweet)
    tweet = sub(r'(\S*?)(\w)\2+\b', word_end_repeat, tweet)
    tweet = tweet.split()
    return ' '.join(tweet)


def down_case(word):
    if word.group() is not None:
        lower = word.group().lower()
        result = lower + " <ALLCAPS> "
        return result


def replace_hashtag(hashtag):
    hashtag = hashtag.group()
    if hashtag is not None:
        hashtag_body = hashtag[1:]
        result = ""
        if hashtag_body.upcase == hashtag_body:
            result = f"<HASHTAG> #{hashtag_body} <ALLCAPS>"
        else:
            result = " ".join(["<HASHTAG>"] + hashtag_body.split(" / (?=[A-Z]) /)"))
        return result


###Labels Preprocessing
def encoder(label):
    if label == 'happiness':
        return 0
    elif label == 'neutral':
        return 1
    return 2


def decoder(label):
    if label == 0:
        return 'happiness'
    elif label == 1:
        return 'neutral'
    return 'sadness'


def preprocess_labels(labels):
    data_labels = labels.apply(lambda x: encoder(x))
    return np.array(data_labels)


def run_preprocessing():
    train = pd.read_csv('TrainEmotions.csv')
    test = pd.read_csv('TestEmotions.csv')

    train['clean_tweet'] = np.zeros(train.shape[0])
    for i, tweet in enumerate(train.content.values):
        clean_tweet = preprocess_glove_line(tweet)
        train.iloc[i, 2] = clean_tweet

    train.to_csv('files_glove/train_glove')
    ## create vocabulary from train
    vocab_to_int, int_to_vocab = create_vocabulary(all_text = train['clean_tweet'].values)
    pd.DataFrame.from_dict(vocab_to_int, orient='index').to_csv('files_glove/vocab_to_int.csv',
                                                                index=True, header=True)

    vocab_words = vocab_to_int.keys()
    max_tweet = train.loc[train[['clean_tweet']].apply(lambda x: x.str.len(), axis=1).idxmax()]
    max_tweet_len = max_tweet['clean_tweet'].str.len().values[0]
    ##apply final preprcoesing on train
    final_train  = []
    for i, clean_tweet in enumerate(train.clean_tweet.values):
        token_tweet = tokenize_line(clean_tweet, vocab_to_int)
        final_tweet = pad_features(token_tweet, seq_length=max_tweet_len)
        final_train.append(final_tweet)

    final = np.array(final_train)
    np.save('files_glove/Train_data', final)
    train_labels = preprocess_labels(train.emotion)
    np.save('files_glove/Train_labels', train_labels)

    ## apply preprocessing on test
    test['clean_tweet'] = np.zeros(test.shape[0])
    final_test = []
    for i, tweet in enumerate(test.content.values):
        clean_tweet = preprocess_glove_line(tweet)
        clean_tweet_filter = [c for c in clean_tweet.split() if c in vocab_words]
        clean_tweet_filter = ' '.join(clean_tweet_filter)
        test.iloc[i, 2] = clean_tweet_filter
        token_tweet = tokenize_line(clean_tweet_filter, vocab_to_int)
        final_tweet = pad_features(token_tweet, seq_length=max_tweet_len)
        final_test.append(final_tweet)

    final = np.array(final_test)
    np.save('files_glove/Test_data', final)
    test.to_csv('files_glove/test_glove')


#run_preprocessing()
# create weights matrix
# vocab = pd.read_csv('files_glove/vocab_to_int.csv')
# glove = pd.read_csv('glove.csv', header = None)
# target_vocab = vocab.iloc[:, 0].values
# weights_matrix = create_weights(target_vocab , glove)
# np.save('files_glove/weight_mat.npy', weights_matrix)
