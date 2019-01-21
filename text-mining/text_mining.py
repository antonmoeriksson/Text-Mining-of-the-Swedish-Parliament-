import imageio as imageio
import numpy as np
import pydotplus as pydotplus

import urllib.request

import itertools
import seaborn as sn

from collections import Counter
import lxml
import bs4
from bs4 import BeautifulSoup

import spacy
from spacy.lang.sv import *
from spacy.lang.sv import stop_words

import time
import re
import pickle
import csv
import io

import pandas as panda
import matplotlib.pyplot as plt
import datetime

import os

from scipy import misc
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB



def read_file(file_name):
    """

    :param file_name:
    :return:
    """
    with open(file_name) as file:
        reader = csv.reader(file)
        input = list(reader)

    return input

def string_column_to_int_column(panda_dataframe):
    panda_dataframe["född"] = panda_dataframe["född"].astype(int)
    panda_dataframe["datum"] = panda.to_datetime(panda_dataframe['datum'])
    panda_dataframe["talartid"] = panda_dataframe["talartid"].astype(int)
    panda_dataframe["tecken"] = panda_dataframe["tecken"].astype(int)
    panda_dataframe["aktiviteter"] = panda_dataframe["aktiviteter"].astype(int)

    return panda_dataframe


def create_pandas_dataframe(file_content):
    index = list(range(0, len(file_content) - 1))
    col = file_content[0]

    return panda.DataFrame(file_content[1:], index=index, columns=col)


def explore_dataset():
    file_content = read_file("Sagtochgjort.csv")

    panda_dataframe = create_pandas_dataframe(file_content)
    panda_dataframe = string_column_to_int_column(panda_dataframe)
    # print(panda_dataframe)
    # print(panda_dataframe["datum"])
    # print(panda_dataframe.describe())


    speeches = panda_dataframe[panda_dataframe.dokumenttyp == 'anf']
    speeches_by_years = list()
    year_10_11 = speeches[speeches.riksmöte == '2010/11']
    year_11_12 = speeches[speeches.riksmöte == '2011/12']
    year_12_13 = speeches[speeches.riksmöte == '2012/13']
    year_13_14 = speeches[speeches.riksmöte == '2013/14']
    year_14_15 = speeches[speeches.riksmöte == '2014/15']
    year_15_16 = speeches[speeches.riksmöte == '2015/16']
    year_16_17 = speeches[speeches.riksmöte == '2016/17']
    dummy_data = speeches[speeches.talartid > 500]['parti']
    print(dummy_data)
    year_16_17.groupby("parti")['talartid'].count().plot(kind='bar')
    # plt.plot(panda_dataframe["talare"], panda_dataframe["tecken"])
    plt.show()
    return None


def desiction_tree_classefaying():
    file_content = read_file("Sagtochgjort.csv")

    panda_dataframe = create_pandas_dataframe(file_content)
    panda_dataframe = string_column_to_int_column(panda_dataframe)


    training_data, test_data = train_test_split(panda_dataframe, test_size=0.25)
    print("Size of Traning data is: " + str(len(training_data)) + "\nSize of the Testing data is: " + str(len(test_data)))

    classifier = DecisionTreeClassifier(min_samples_split=1000)
    features = ["född", "talartid", "tecken", "aktiviteter"]

    x_training_data = training_data[features]
    y_traning_data = training_data['parti']


    x_test_data = test_data[features]
    y_test_data = test_data['parti']

    desiction_tree = classifier.fit(x_training_data, y_traning_data)

    file = io.StringIO()
    export_graphviz(desiction_tree, out_file=file, feature_names=features)
    pydotplus.graph_from_dot_data(file.getvalue()).write_png("dec_tree.png")
    pic = imageio.imread("dec_tree.png")
    #plt.imshow(pic)

    y_predict = classifier.predict(x_test_data)

    acc = accuracy_score(y_test_data, y_predict)
    print(acc*100)
    return None

def npl_for_swedish_parlament():
    nlp = Swedish()
    my_file = open("test-motion.txt").read()
    my_file = my_file.replace('\n', ' ')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    nlp_file = nlp(my_file)

    list_of_tokens_in_sent = list()
    for i, sentenace in enumerate(nlp_file.sents):
        #print(str(i) + ": " + str(sentenace))
        list_of_tokens_in_sent.append([token.text.lower() for token in sentenace if not token.is_stop and token.is_alpha])
    print(list_of_tokens_in_sent)




    return None

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
    app_url_set = set()
    party_dict = {}
    #print(len(content))
    for c in content:
        #print(c)
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
            #print("Part is : " + str(party[i]))
            party_dict[party[i]] += 1

    for key in party_dict:
        party_dict[key] /= 2
        party_dict[key] = int(party_dict[key])


    print(app_url_list[0])
    print(app_url_list[1])

    # print("Len of app url list" + str(len(app_url_list)))
    # for url in app_url_list:
    #     app_url_set.update(url)
    #
    # #print(len(app_url_set))
    # app_url_list = list(app_url_set)
    #
    # #print(len(app_url_list[0]))
    # #print(len(app_url_list[1]))

    url_content_list = list()
    url_to_text_dict = {}
    i = 0
    # content = urllib.request.urlopen('https://play.google.com/' + app_url_list[0]).read().decode('utf-8')
    # print(content)
    # url_to_text_dict[app_url_list[0]] = re.findall(desc_re, content)
    i = 0
    for q, elem in enumerate(app_url_list):
        print("#" + str(q))
        for i, url in enumerate(elem):
            url_content = urllib.request.urlopen('https:' + url).read().decode('utf-8')
            #print(url_content_list[-1])
            #print(type(url_content_list[-1]))
            # soup = BeautifulSoup(url_content_list[-1], 'lxml')
            # text = soup.get_text().replace('\n', ' ')
            # text = text.replace('-->', '')
            # text = text.replace('  ', '')
            # text = text.lower()

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



        #print(text)

        #url_to_text_dict[url] = text

        #print(url_content_list[i])

        #print(url_to_text_dict[url])
        #prit(i)
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
    #url_to_text_dict = pickle_dict["url_to_text_dict"]
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
            #print(str(i) + ": " + str(sentenace))
            list_of_tokens_from_text.append(
                [token.text.lower() for token in sentenace if token not in stop_words.STOP_WORDS and token.is_alpha])
        clean_text_list.append(list_of_tokens_from_text[0])


    print("Number of texts = " + str(len(clean_text_list)))
#    print(url_to_tokens_dict['//data.riksdagen.se/dokument/H6022610.html'])

    trans_vector = TfidfVectorizer(stop_words="swedish", min_df=0.04, max_df=0.5)
    #array_of_texts = np.empty(len(url_to_text_dict))





    flat_party_list = (list(itertools.chain(*party_list)))
    flat_party_list = flat_party_list[1::2]
    print(len(flat_party_list))
    print(flat_party_list)
    y_axies = panda.DataFrame(flat_party_list, columns=["labels"])
    print(y_axies.head(5))
    print("Len of party list [0] = " + str(party_list))
    #print(party_list[0])
    #print(y_axies.head(40))
    #print((len(list_of_texts[0])))
    #print((len(list_of_texts[1])))

    cont_vec = CountVectorizer()


    #print(features)
    #print(fitting.shape)
    flat_text_list = list()
    for text_list in clean_text_list:
        flat_text_list.append(' '.join(text_list))
    print("len of flat text list " + str(flat_text_list[0]))



    data_frame = panda.DataFrame(flat_text_list, columns=["text"])

    print(data_frame.head(6))
    print()
    training_data_X, test_data_X,training_data_Y, test_data_Y= train_test_split(data_frame,y_axies, test_size=0.80,)
    #training_data_Y, test_data_Y = train_test_split(y_axies, test_size=0.80)

    print("Afetr splitxuu")
    #print(type(training_data_X))
    fitting = cont_vec.fit_transform(training_data_X.text)
    #print(fitting)
    print("Beeeer")
    #print(str(fitting.shape))
    print(training_data_Y.shape)
    print(training_data_X.shape)

    print(test_data_Y.shape)
    print(test_data_X.shape)
    features = cont_vec.get_feature_names()
    #print(str(features))

    #print(data_frame.shape)
    #print(data_frame.head(3))

    print(training_data_X.shape)
    print(training_data_Y.shape)

    #print(test_data_X.shape)
    #print(test_data_Y.shape)


    #gaussian = GaussianNB()
    #gaussian_fitting =gaussian.fit(fitting.toarray(), training_data_Y)
    #gaussian_predtion = gaussian_fitting.predict(test_data_X)
    #print(type(gaussian_fitting))
    print("Shape of  test_data_Y = " + str(test_data_Y.shape))
    print("Shape of  fitting = " + str(fitting.shape))


    multiNB = MultinomialNB()
    multiNB_fitting = multiNB.fit(fitting, training_data_Y.labels)

    y_data = cont_vec.transform(test_data_X.text)
    print(test_data_X.shape)
    print(y_data.shape)

    multiNB_predictioon = multiNB_fitting.predict(y_data)
    #print(str(multiNB_predictioon.classes_))

    print("End")

    print(y_data)
    print("Accercy of the fuction {} is {}".format("Multinomila",accuracy_score(test_data_Y, multiNB_predictioon)))
    print("F1of the fuction {} is {}".format("Multinomila",f1_score(test_data_Y, multiNB_predictioon, average='weighted')))
    print("Recallof the fuction {} is {}".format("Multinomila",recall_score(test_data_Y, multiNB_predictioon,average='weighted')))
    print("Persction of the fuction {} is {}".format("Multinomila",precision_score(test_data_Y, multiNB_predictioon,average='weighted')))

    train = panda.DataFrame([training_data_X.text, training_data_Y.labels])
    print(train.head(2))
    #train.columns = ["text", "lables"]
    #train['cat'] = train.labels.factorize()[0]
    #cat_id = train[['label', 'cat']].drop_duplicates().sort_values('cat').reset_index(drop=True)


    print(confusion_matrix(test_data_Y.labels, multiNB_predictioon))
    cm = confusion_matrix(test_data_Y.labels, multiNB_predictioon)
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
    #desiction_tree_classefaying()
    #npl_for_swedish_parlament()
    #get_data_from_urls()
    #concatenate_collected_data()
    pre_process_data()

if __name__ == '__main__':
    main()