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
    # names.remove("BOTH.")
    # names.remove("ALL.")

    return names


def cleanUp_list(text, characters):
    stringList = text.splitlines()
    cleanList = list()
    for string in stringList:
        if len(string) > 3:
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

    for row in dialogue_list:
        s = row# your string here
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

    print(dialogue_df)
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

    print(emotions_df)

    return play_df, emotions_df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # dollhouse = load_etext(15492, refresh_cache=True)
    book_id = 2542

    text = strip_headers(load_etext(book_id))
    names = extract_characters(text)
    stringList = cleanUp_list(text, names)

    df = create_data_frame(stringList, names)
    df, emotions_df = find_sentiment(df)
    cwd = os.getcwd()
    path = cwd + f'/play_analysis_{book_id}.csv'
    em_path = cwd + f'/play_emotions_{book_id}.csv'

    path_xml = cwd + f'/play_analysis_{book_id}.xml'
    em_path_xml = cwd + f'/play_emotions_{book_id}.xml'

    df.to_csv(path)
    emotions_df.to_csv(em_path)

    # df.to_xml(path_xml)
    # emotions_df.to_xml(em_path_xml)


    print(df)

