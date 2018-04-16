from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from TSA.Preproc import Preproc
lemmatizer = WordNetLemmatizer()


def convert_tag(tag):
    if tag.startswith("J"):
        return wn.ADJ
    elif tag.startswith("N"):
        return wn.NOUN
    elif tag.startswith("R"):
        return wn.ADV
    elif tag.startswith("V"):
        return wn.VERB
    return None


def clean_text(filepath, filename):
    pp = Preproc.Preproc()
    pp.loadCsv(filepath, filename)
    pp.remove_html_encode()
    return pp.get_twitter_df()


def swn_classifier(tweet):
    sentiment = 0.0
    tokens_count = 0

    sentinized_tweet = sent_tokenize(tweet)

    for sentence in sentinized_tweet:
        # Part of Speach tag
        tagged_tweet = pos_tag(word_tokenize(sentence))

        for word, tag in tagged_tweet:
            wn_tag = convert_tag(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue

            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue

            synset = wn.synsets(lemma, pos=wn_tag)
            # If not a synonym skip this word.
            if not synset:
                continue

            synset = synset[0]
            swn_synset = swn.senti_synset(synset.name())

            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            tokens_count += 1

    # if not word match the lexicon default to negative
    if not tokens_count:
        return 0

    if sentiment >= 0:
        return 1

    # negative sentiment.
    return 0


def main():
    # for STS dataset
    # df = clean_text("../datasets/STS/", "STS.csv")

    # for SemEval dataset
    df = clean_text("../datasets/SemEval/4A-English/", "SemEval.csv")

    # split data into test 80% train, 20%
    X_train, X_test, y_train, y_test = train_test_split(df.tweet, df.lable, test_size=0.2, random_state=0)
    pred_y = [swn_classifier(tweet) for tweet in X_test]
    print(accuracy_score(y_test, pred_y))


if __name__ == '__main__':
    main()