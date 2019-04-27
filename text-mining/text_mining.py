import imageio as imageio
import pydotplus as pydotplus

import urllib.request

import itertools

from bs4 import BeautifulSoup

from spacy.lang.sv import *
from spacy.lang.sv import stop_words

import re
import pickle
import csv
import io

import pandas as panda
import gensim
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models

def pre_process_data(text):
    """
    Tokenization the text and removes stopwords.
    :param text:
    :return: [tokens]
    """
    ploictical_common_words = {"motion"}
    result = list()
    for token in simple_preprocess(text):
        if not (token in STOPWORDS or token in ploictical_common_words):
            result.append(token)
    return result


def get_data_from_urls():
    url_list = []

    problem_index = [7,8,10,11,12,22,38,42]

    woriking_pages = list(range(1,7)) + list(range(13,21)) + list(range(23,37)) + list(range(43,100))
    pages = []
    for i in woriking_pages:
        url_list.append('https://data.riksdagen.se/dokumentlista/?u17=22&sok=&doktyp=mot&rm=&from=&tom=&ts=&bet=&tempbet'
                        '=&nr=&org=&iid=&parti=S%2cM%2cL%2cKD%2cV%2cSD%2cC%2cMP%2cNYD%2c-&webbtv'
                        '=&talare=&exakt=&planering=&sort=rel&sortorder=desc&rapport=&utformat=json&a=s&p=' + str(i))

    content = []
    for url in url_list:
        content.append(urllib.request.urlopen(url).read().decode('utf-8'))

    appreg = r'(\/\/data.riksdagen.se\/dokument\/.+.html)'
    party_regex = r'[A-Z]+\)'
    party_regex = re.compile(party_regex)
    appre = re.compile(appreg)

    app_url_list = []
    party_list = list()
    party_dict = {}
    for c in content:
        app_url_list.append(re.findall(appre, c))
        party_list.append(re.findall(party_regex, c))

    for j, url in enumerate(app_url_list):
        print("The len of app URL list {}  and URL has the len {}".format(j, len(url)))

    print("The len of PARTYLIST [0] {}  and and PARTY LIST[0] {}".format(len(party_list[0]), len(party_list[1])))

    corrupted_indexes = list()
    for a, party in enumerate(party_list):
        if len(party) > 40:
            corrupted_indexes.append(a)
            print("WARNING LEN PARTY > 20 at index = " + str(a))
        for i, elem in enumerate(party):
            party[i] = "(" + elem
            if party[i] not in party_dict:
                party_dict[party[i]] = 0
            party_dict[party[i]] += 1

    for key in party_dict:
        party_dict[key] /= 2
        party_dict[key] = int(party_dict[key])

    url_content_list = list()
    url_to_text_dict = {}
    i = 0
    for q, elem in enumerate(app_url_list):
        print("#" + str(q))
        for i, url in enumerate(elem):
            url_content = urllib.request.urlopen('https:' + url).read().decode('utf-8')
            soup = BeautifulSoup(url_content, 'lxml')  # create a new bs4 object from the html data loaded
            for script in soup(["script", "style"]):  # remove all javascript and stylesheet code
                script.extract()
            # get text
            text = soup.get_text()
            # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)

            regex = re.compile(r'[^a-öA-Ö ]')
            # First parameter is the replacement, second parameter is your input string
            text = regex.sub('', text.lower())

            url_content_list.append(text)
    print("url conetent list = " + str(len(url_content_list)))

    total_dict = 0
    for key in party_dict:
        total_dict += party_dict[key]
    total_list = 0
    for elem in party_list:
        total_list += len(elem)

    print("The total lenght in the Party LIST is: {}\n The total lenght in the Party DICTis: {}".format(total_list, total_dict))


    pickle_dict = {}
    pickle_dict["url_content_list"] = url_content_list
    pickle_dict["party_dict"] = party_dict
    pickle_dict["party_list"] = party_list
    pickle_dict["url_to_text_dict"] = url_to_text_dict
    pickle.dump(pickle_dict, open("test_save.p", "wb"))

    return None


def pre_process_data():
    pickle_dict = pickle.load(open("test_save.p", "rb"))
    party_dict = pickle_dict["party_dict"]
    party_list = pickle_dict["party_list"]
    url_content_list = pickle_dict["url_content_list"]
    clean_text_list = list()
    print(party_dict)


    url_to_tokens_dict = {}
    nlp = Swedish()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    print(stop_words.STOP_WORDS)

    for text in url_content_list:
        nlp_file = nlp(text)
        list_of_tokens_from_text = list()
        for i, sentenace in enumerate(nlp_file.sents):
            list_of_tokens_from_text.append(
                [token.text.lower() for token in sentenace if token not in stop_words.STOP_WORDS and token.is_alpha])
        clean_text_list.append(list_of_tokens_from_text[0])


    print("Number of texts = " + str(len(clean_text_list)))






    flat_party_list = (list(itertools.chain(*party_list)))
    flat_party_list = flat_party_list[1::2]
    print(len(flat_party_list))
    print(flat_party_list)
    y_axies = panda.DataFrame(flat_party_list, columns=["labels"])
    print(y_axies.head(5))
    print("Len of party list [0] = " + str(party_list))


    flat_text_list = list()
    for text_list in clean_text_list:
        flat_text_list.append(' '.join(text_list))
    print("len of flat text list " + str(flat_text_list[0]))



    data_frame = panda.DataFrame(flat_text_list, columns=["text"])
    df = panda.DataFrame(data=[(y_axies['labels'], data_frame['text'])], columns=['party', 'text'])
    #  lda_model = lda_baseline(df)

    #  Here we have Two Pandas data frame,
    #  'data_frame' and 'y_axes' here we should begin to create a bag of word approach.
    training_data_X, test_data_X,training_data_Y, test_data_Y= train_test_split(data_frame,y_axies, test_size=0.80,)

    return training_data_X, test_data_X, training_data_Y, test_data_Y


def create_lda_model(df):
    processed_documents = df["text"].map(pre_process_data)

    # Creates as dictionary containing words and word counts, and filtering out extremes.
    dictionary = corpora.Dictionary(processed_documents)
    dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=100000)

    # Creates a Bag of Words linking the words in each document to it's word count.
    bag_of_words = [dictionary.doc2bow(doc) for doc in processed_documents]

    # Create and applies a Term frequency–inverse document frequency model to the bag of words.
    tfidf = models.TfidfModel(bag_of_words)
    corpus_tfidf = tfidf[bag_of_words]

    # Create LDA model from the TF-IDF model.
    return gensim.models.LdaMulticore(corpus_tfidf, num_topics=8, id2word=dictionary,
                                                 workers=4, minimum_probability=0.0)


def present_results(test_data_x, test_data_y, training_data_x, training_data_y):
    # Init all different vectorizers.
    bow_vectorizer = CountVectorizer()
    bigram_vecorizer = CountVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf_vectorizer = TfidfVectorizer()

    # Fit all the vectorizers on the training data.
    bow_fitting = bow_vectorizer.fit_transform(training_data_x.text)
    bigram_fitting = bigram_vecorizer.fit_transform(training_data_x.text)
    tfidf_fitting = tfidf_vectorizer.fit_transform(training_data_x.text)

    #  Init machine learning models to use.
    multi_nb = MultinomialNB()

    #  Training of the machine learning models on the training data.
    multi_nb_bow_fitting = multi_nb.fit(bow_fitting, training_data_y.labels)
    multi_nb_bigram_fitting = multi_nb.fit(bigram_fitting, training_data_y.labels)
    multi_nb_tfidf_fitting = multi_nb.fit(tfidf_fitting, training_data_y.labels)

    #  Transform the test test data to the correct form.
    y_data_bow = bow_vectorizer.transform(test_data_x.text)
    y_data_bigram = bigram_vecorizer.transform(test_data_x.text)
    y_data_tfidf = tfidf_vectorizer.transform(test_data_x.text)

    print(test_data_x.shape)
    print(y_data_bow.shape)

    #  Performers the prediction with the models.
    multi_nb_bow_prediction = multi_nb_bow_fitting.predict(y_data_bow)
    multi_nb_bigram_prediction = multi_nb_bigram_fitting.predict(y_data_bigram)
    multi_nb_tfidf_prediction = multi_nb_tfidf_fitting.predict(y_data_tfidf)

    # Analizes the models predictions.
    print("Accercy of the fuction {} is {}".format("Multinomila", accuracy_score(test_data_y, multi_nb_bow_prediction)))
    print("  the fuction {} is {}".format("Multinomila",
                                             f1_score(test_data_y, multi_nb_bow_prediction, average='weighted')))
    print("Recall of the fuction {} is {}".format("Multinomila",
                                                 recall_score(test_data_y, multi_nb_bow_prediction, average='weighted')))
    print("Persction of the fuction {} is {}".format("Multinomila", precision_score(test_data_y, multi_nb_bow_prediction,
                                                                                    average='weighted')))
    train = panda.DataFrame([training_data_x.text, training_data_y.labels])
    print(train.head(2))
    # train.columns = ["text", "lables"]
    # train['cat'] = train.labels.factorize()[0]
    # cat_id = train[['label', 'cat']].drop_duplicates().sort_values('cat').reset_index(drop=True)
    print(confusion_matrix(test_data_y.labels, multi_nb_bow_prediction))
    cm = confusion_matrix(test_data_y.labels, multi_nb_bow_prediction)
    plt.matshow(cm)
    plt.title("TEST")
    plt.show()


def concatenate_collected_data():
    great_url_to_text_dict = {}
    great_party_dict = {}
    great_party_list = list()
    for elem in [1,2,3,8,9,10,15] :
        #if i != 4 and i != 5 and i !=6 and i != 11:
        print("#" + str(elem))

        pickle_dict = pickle.load(open("save_" + str(elem) + ".p", "rb"))

        local_url_to_text_dict = pickle_dict["url_to_text_dict"]
        great_party_list += pickle_dict["party_list"]
        local_party_dict ={}
        local_party_dict.update(pickle_dict["party_dict"])
        print()
        print(local_party_dict.keys())

        for key, value in local_party_dict.items():
            print(str(value))
            if key not in great_party_dict.keys():
                great_party_dict[key] = local_party_dict[key]
            else:
                great_party_dict[key] += local_party_dict[key]

        for key in local_url_to_text_dict:
            if key not in great_url_to_text_dict.keys():
                great_url_to_text_dict[key] = local_url_to_text_dict[key]
                print("Not duplicate")
            else:
                print("DUPLICATE")

        print("LOCAL URL TO TEXT: "+str(len(pickle_dict["url_to_text_dict"])))
        print("LOCAL PARTY DICT: " + str(len(pickle_dict["party_dict"])))
        print("LOCAL PARTY LIST: " + str(len(pickle_dict["party_list"])))

        print("The len of URL TO TEXT: " + str(len(great_url_to_text_dict)))
        print("The len of PARTY DICT: " + str(len(great_party_dict)))
        print("The len of PARTY LIST: " + str(len(great_party_list)))


    print(str(great_party_dict))
    pickle_dict2 = {}
    pickle_dict2["party_dict"] = great_party_dict
    pickle_dict2["party_list"] = great_party_list
    pickle_dict2["url_to_text_dict"] = great_url_to_text_dict
    pickle.dump(pickle_dict2, open("great_save.p", "wb"))


def main():
    """

    :return:
    """
    print("Let's go")
    #desiction_tree_classefaying()
    #npl_for_swedish_parlament()
    #get_data_from_urls()
    #concatenate_collected_data()
    training_data_x, test_data_x, training_data_y, test_data_y = pre_process_data()
    present_results(test_data_x, test_data_y, training_data_x, training_data_y)


if __name__ == '__main__':
    main()