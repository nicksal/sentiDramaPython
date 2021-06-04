# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers

import re
import os
import pandas as pd
import text2emotion as t2e
from flair.models import TextClassifier
from flair.data import Sentence


def extract_characters(text):
    # ^[A-Z]+\s?[A-Z]*\.$
    namesDuplicates = re.findall(r'[A-Z]+[\\.$]', text)
    names = list(set(namesDuplicates))

    r = re.compile('/SCENE.*/i')
    x = re.compile('^I.*I.$')
    z = re.compile('I.')
    zx = re.compile('V.')
    y = re.compile('^V.*I.$')
    v = re.compile('^I.*V.$')

    filtered_scene = list(filter(r.match, names))
    filtered_act = list(filter(x.match, names))
    filtered_first_act = list(filter(z.match, names))
    filtered_act_v = list(filter(zx.match, names))
    filtered_latin_actV = list(filter(y.match, names))
    filtered_latin_actIV = list(filter(v.match, names))

    names = [x for x in names if x not in filtered_act]
    names = [x for x in names if x not in filtered_scene]
    names = [x for x in names if x not in filtered_first_act]
    names = [x for x in names if x not in filtered_latin_actV]
    names = [x for x in names if x not in filtered_latin_actIV]
    names = [x for x in names if x not in filtered_act_v]
    names.remove("SCENE.")
    # TODO: Tweaks for generic case
    # names.remove("BOTH.")
    # names.remove("ALL.")

    return names


def cleanUp_list(text, characters):
    stringList = text.splitlines()
    cleanList = list()
    for string in stringList:
        if len(string) > 3:
            # TODO: Tweaks for generic case
            # res = any(map(string.__contains__, characters))
            # if res:
            #     names = re.findall(r'[A-Z]+[\\.$]', string)
            #     rest = string.replace(names[0], '')
            #     cleanList.append(names[0])
            #     cleanList.append(rest)
            # else:
            cleanList.append(string)
    cleanList.append(characters[0])
    return cleanList


def create_data_frame(dialogue_list, characters):

    columnNames = ["id", "act", "person", "dialogue"]
    dialogue_df = pd.DataFrame(columns=columnNames)

    acts = []
    for row in dialogue_list:
        if "ACT" in row:
            x = re.search('\.$', row)
            if (x == None):
                acts.append(row)

    acts = list(set(acts))

    id = 0
    act = ''
    actIncremental = 0
    person = ''
    dialoge = ''

    startTrackingDialogue = False
    # Logic goes as:
    # Iterate each row
    #   Find the first row that matches a person
    #   Save the next row in a local var
    #   Until we find the row that matches the next person
    #   when we find the next person we store the id, actNumber, dialogue and person in our dataframe row
    # Repeat till EOF

    for row in dialogue_list:
        s = row# your string here
        # TODO: Tweaks for generic case re.sub('\[.*?\].', '', s)
        row = re.sub('_\[.*?\]_.', '', s)
        if row in acts:
            actIncremental += 1
            act = f'ACT {actIncremental}'
        else:
            res = any(map(row.__contains__, characters))
            if res:
                if person == '':
                    person = row
                    startTrackingDialogue = True
                else:
                    # columnNames = ["id", "act", "person", "dialogue", "sentimentScore", "sentimentLabel"]
                    dialogue_df.loc[id] = [id, act, person, dialoge]
                    id += 1
                    person = row
                    dialoge = ''
            else:
                if startTrackingDialogue:
                    dialoge += row


    return dialogue_df


def find_sentiment(play_df):
    classifier = TextClassifier.load('en-sentiment')
    listOfEmotions = list()

    for index, row in play_df.iterrows():
        dialogue = row['dialogue']
        # emotion voc based
        emotions = t2e.get_emotion(dialogue)
        # sentiment ml based
        sentence = Sentence(dialogue)
        classifier.predict(sentence)
        # Usage
        if len(sentence.labels) > 0:
            play_df.at[index, 4] = sentence.labels[0].value
            play_df.at[index, 5] = sentence.labels[0].score


        sentence.labels.clear()
        listOfEmotions.append(emotions)

    emotions_df = pd.DataFrame.from_dict(listOfEmotions)


    return play_df, emotions_df

if __name__ == '__main__':
    # dollhouse = load_etext(15492, refresh_cache=True)
    # Set the book id in order to be retrieved
    book_id = 2542

    # load_etext loads the book with book_id from gutenberg.com
    # strin_headers removes the licensing information
    text = strip_headers(load_etext(book_id))

    # start the processing by extracting the persons of the play
    # various heuristics rules applied
    # might need some tweaks in order to play in a generic case
    # search for #TODO

    names = extract_characters(text)

    # get the list seperating each row and extracting the dialogueId, act, dialogueString
    stringList = cleanUp_list(text, names)
    # create a pandas dataframe in order to make our life easier manipulating the data
    df = create_data_frame(stringList, names)
    # run the sentiment check for each dialogue
    df, emotions_df = find_sentiment(df)

    # outputs
    cwd = os.getcwd()
    path = cwd + f'/play_analysis_{book_id}.csv'
    em_path = cwd + f'/play_emotions_{book_id}.csv'

    path_xml = cwd + f'/play_analysis_{book_id}.xml'
    em_path_xml = cwd + f'/play_emotions_{book_id}.xml'

    df.to_csv(path)
    emotions_df.to_csv(em_path)

    # Needs pandas 1.3 to work! (as of 04/06/2021 still in beta)
    # df.to_xml(path_xml)
    # emotions_df.to_xml(em_path_xml)


