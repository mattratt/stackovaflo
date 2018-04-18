import sys
import logging
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
pd.set_option('display.width', 240)


POST_FIELDS = ['Id', 'CreationDate', 'OwnerUserId', 'Score', 'CommentCount']
QUESTION_FIELDS = POST_FIELDS + ['ViewCount', 'AnswerCount', 'FavoriteCount', 'AcceptedAnswerId']
ANSWER_FIELDS = POST_FIELDS + ['ParentId']
# skipped: PostTypeId, Body, Title, Tags, LastEditorUserId, LastEditorDisplayName, LastEditDate,
#          LastActivityDate

USER_FIELDS = ['Id', 'Reputation', 'CreationDate', 'Location', 'Views', 'UpVotes', 'DownVotes',
               'Age', 'AccountId']  #zzz which id field to use?  Id or AccountId?
# skipped: DisplayName, LastAccessDate, WebsiteUrl, AboutMe


def parse_posts(infile):
    questions = []
    answers = []
    accepted_answer_ids = set()
    for i, line in enumerate(infile):
        if i % 10000 == 0:
            logging.debug("\t{}\t{} questions,\t{} answers".format(i, len(questions), len(answers)))

        try:
            rec = ET.fromstring(line)
        except ET.ParseError as err:
            logging.debug("can't parse line {}: {}".format(i, line.strip()))
            continue

        if rec.tag != 'row':
            continue

        # this one's special since it's calculated
        body_len = len(rec.attrib['Body']) if 'Body' in rec.attrib else None

        if rec.attrib['PostTypeId'] == '1':  # question
            vals = [ rec.attrib.get(attr) for attr in QUESTION_FIELDS ] + [body_len]
            accepted_answer_ids.add(rec.attrib.get('AcceptedAnswerId'))
            questions.append(tuple(vals))

        else:  # answer
            accepted = rec.attrib['Id'] in accepted_answer_ids  # if we come across the answer
                                                                # first, this will break
            vals = [ rec.attrib.get(attr) for attr in ANSWER_FIELDS ] + [body_len, accepted]
            answers.append(tuple(vals))

    logging.info("creating DataFrame for {} questions".format(len(questions)))
    index_vals = get_index_vals(questions, QUESTION_FIELDS, 'Id', int)
    question_df = pd.DataFrame(questions, index=index_vals, columns=QUESTION_FIELDS+['Length'],
                               dtype=np.int64)

    logging.info("creating DataFrame for {} posts".format(len(answers)))
    index_vals = get_index_vals(answers, ANSWER_FIELDS, 'Id', int)
    answer_df = pd.DataFrame(answers, index=index_vals, columns=ANSWER_FIELDS+['Length', 'Accepted'],
                             dtype=np.int64)

    return question_df, answer_df


def get_index_vals(tups, col_names, index_col_name, index_col_type=str):
    p = col_names.index(index_col_name)
    return [index_col_type(t[p]) for t in tups]


def parse_users(infile, selects=None):
    if selects is not None:
        selects = set(selects)
        logging.debug("selecting from {} unique users".format(len(selects)))

    users = []
    reject_count = 0
    for i, line in enumerate(infile):
        if i % 10000 == 0:
            logging.debug("\t{}\t{} users,\t{} rejects".format(i, len(users), reject_count))
        try:
            rec = ET.fromstring(line)
        except ET.ParseError as err:
            logging.debug("can't parse line {}: {}".format(i, line.strip()))
            continue

        if rec.tag != 'row':
            continue

        if (selects is not None) and (rec.attrib['Id'] not in selects):
            reject_count += 1
            continue

        users.append(tuple(rec.attrib.get(attr) for attr in USER_FIELDS))

    logging.info("creating DataFrame for {} users".format(len(users)))
    index_vals = get_index_vals(users, USER_FIELDS, 'Id', int)
    return pd.DataFrame(users, index=index_vals, columns=USER_FIELDS, dtype=np.int64)


####################################

if __name__ == '__main__':

    # parse questions, answers, users
    USAGE = sys.argv[0] + " postfile userfile"  # e.g. parse_stack_overflow.py Posts.xml Users.xml
    if len(sys.argv) < 3:
        sys.exit(" usage: " + USAGE)

    with open(sys.argv[1], 'r') as infile:
        quest_df, ans_df = parse_posts(infile)
        print quest_df.head(), "\n", quest_df.dtypes
        print ans_df.head(), "\n", ans_df.dtypes

    with open(sys.argv[2], 'r') as infile:
        user_df = parse_users(infile)
        print user_df.head(), "\n", user_df.dtypes

    # add answer aggregs to questions, then join user table
    logging.info("aggreg answer cols")
    aggs = {'Score': {'mean_Score': 'mean'},
            'CommentCount': {'mean_CommCount': 'mean'},
            'Length': {'mean_Length': 'mean'}}
    answer_aggregs_df = ans_df.groupby('ParentId').agg(aggs)
    answer_aggregs_df.columns = answer_aggregs_df.columns.droplevel(0)
    # print answer_aggregs_df.head(), "\n", answer_aggregs_df.dtypes

    logging.info("joining answer cols to questions")
    quest_df = quest_df.join(answer_aggregs_df, rsuffix='_answer')
    # print "quest joined:\n", quest_df.loc[quest_df['Id'] == key]
    # print "quest joined:\n", quest_df.loc[key]
    # print quest_df.head(), "\n", quest_df.dtypes

    # for some reason these is object
    quest_df['OwnerUserId'] = pd.to_numeric(quest_df['OwnerUserId'], downcast='integer')
    user_df['Age'] = pd.to_numeric(user_df['Age'], downcast='integer')

    logging.info("joining users and questions")
    user_question_df = quest_df.join(user_df, on='OwnerUserId', rsuffix='_user')
    logging.debug("joined table has {} rows".format(len(user_question_df)))
    # print user_question_df.head(), "\n", user_question_df.dtypes

    logging.info("joining users and answers")
    user_answer_df = ans_df.join(user_df, on='OwnerUserId', rsuffix='_user')
    logging.debug("joined table has {} rows".format(len(user_answer_df)))

    logging.info("joining answers and questions")
    question_answer_df = ans_df.join(quest_df, on='ParentId', rsuffix='_quest')
    logging.debug("joined table has {} rows".format(len(question_answer_df)))

