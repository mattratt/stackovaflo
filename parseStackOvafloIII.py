import sys
import logging
import datetime
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

import Contingency

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
pd.set_option('display.width', 240)


#  <row Id="6"
# PostTypeId="1"
# AcceptedAnswerId="31"
# CreationDate="2008-07-31T22:08:08.620"
# Score="231"
# ViewCount="15363"
# Body="&lt;p&gt;I have an absolutely positioned &lt;code&gt;div&lt;/code&gt; containing..."
# OwnerUserId="9"
# LastEditorUserId="63550"
# LastEditorDisplayName="Rich B"
# LastEditDate="2016-03-19T06:05:48.487"
# LastActivityDate="2016-03-19T06:10:52.170"
# Title="Percentage width child element in absolutely positioned parent on Internet..."
# Tags="&lt;html&gt;&lt;css&gt;&lt;css3&gt;&lt;internet-explorer-7&gt;"
# AnswerCount="5"
# CommentCount="0"
# FavoriteCount="9" />
#
#  <row Id="12"
# PostTypeId="2"
# ParentId="11"
# CreationDate="2008-07-31T23:56:41.303"
# Score="312"
# Body="&lt;p&gt;Well, here's how we do it on Stack Overflow.&lt;/p&gt;&#xA;&#xA;&lt;pre..."
# OwnerUserId="1"
# LastEditorUserId="1420197"
# LastEditorDisplayName="GateKiller"
# LastEditDate="2014-02-18T14:19:35.770"
# LastActivityDate="2014-02-18T14:19:35.770"
# CommentCount="10"
# CommunityOwnedDate="2009-09-04T13:15:59.820" />
#
POST_FIELDS = ['Id', 'CreationDate', 'OwnerUserId', 'Score', 'CommentCount']
QUESTION_FIELDS = POST_FIELDS + ['ViewCount', 'AnswerCount', 'FavoriteCount', 'AcceptedAnswerId']
ANSWER_FIELDS = POST_FIELDS + ['ParentId']
# skipped: PostTypeId, Body, Title, Tags, LastEditorUserId, LastEditorDisplayName, LastEditDate,
# LastActivityDate

# <row Id="35"
# Reputation="14839"
# CreationDate="2008-08-01T12:43:07.713"
# DisplayName="Greg Hurlman"
# LastAccessDate="2017-11-02T18:27:00.923"
# WebsiteUrl="http://greghurlman.com"
# Location="Washington, DC"
# AboutMe="&lt;p&gt;Solution Architect on the Parature team at Microsoft based in Washington, DC.&lt;br/&gt;&#xA;&lt;br/&gt;&#xA;Coder for 25 years, .Net developer since 2001.&lt;br/&gt;&#xA;&lt;br/&gt;&#xA;Twitter: @justcallme98&lt;br /&gt;&#xA;&lt;br /&gt;&#xA;Xbox Live Gamertag: ExitNinetyEight&lt;/p&gt;&#xA;"
# Views="1392"
# UpVotes="449"
# DownVotes="211"
# Age="39"
# AccountId="26" />
USER_FIELDS = ['Id', 'Reputation', 'CreationDate', 'Location', 'Views', 'UpVotes', 'DownVotes',
               'Age', 'AccountId']
#zzz which id field to use?
# skipped: DisplayName, LastAccessDate, WebsiteUrl, AboutMe


def select_by_date(tag, start_dt, end_dt, infile, outfile, date_attr='CreationDate'):
    mark = '<' + tag + ' '
    good_count = 0
    bad_count = 0
    dt = None
    for i, line in enumerate(infile):
        if i % 100000 == 0:
            logging.debug("\t{} {} kept {}".format(i, dt, good_count))
        if line.lstrip().startswith(mark):
            parsed = ET.fromstring(line)
            if date_attr in parsed.attrib:
                # 2008-07-31T23:55:37.967
                dt = datetime.datetime.strptime(parsed.attrib[date_attr], "%Y-%m-%dT%H:%M:%S.%f")
                if (dt < start_dt) or (end_dt <= dt):
                    bad_count += 1
                    continue
                else:
                    good_count += 1

        outfile.write(line)
    logging.debug("kept {} / {} {} records".format(good_count, good_count+bad_count, tag))


def select_by_value(tag, select_attr, select_vals, infile, outfile):
    mark = '<' + tag + ' '
    good_count = 0
    bad_count = 0
    for i, line in enumerate(infile):
        if i % 100000 == 0:
            logging.debug("\t{} kept {}".format(i, good_count))
        if line.lstrip().startswith(mark):
            parsed = ET.fromstring(line)
            if parsed.attrib.get(select_attr) not in select_vals:
                bad_count += 1
                continue
            else:
                good_count += 1
        outfile.write(line)
    logging.debug("kept {} / {} {} records".format(good_count, good_count+bad_count, tag))


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


def pearson(df_orig, x, y):
    xy_mat = df_orig[[x, y]].dropna().as_matrix()
    # stat, pval = Contingency.stats.pearsonr(xy_mat[:, 0], xy_mat[:, 1])
    stat = Contingency.zscore(xy_mat[:, 0].tolist(), xy_mat[:, 1].tolist())
    pval = Contingency.zscoreP(stat)
    return stat, pval

def pearson_partial(df_orig, x, y, z):
    xyz_mat = df_orig[[x, y, z]].dropna().as_matrix()
    stat, pval = Contingency.partial_corr(xyz_mat[:, 0], xyz_mat[:, 1], xyz_mat[:, [2]], pval=True)
    return stat, pval

def pearson_block(df_orig, x, y, z):
    xyz_mat = df_orig[[x, y, z]].dropna().as_matrix()
    x_lst = xyz_mat[:, 0].tolist()
    y_lst = xyz_mat[:, 1].tolist()
    z_lst = xyz_mat[:, 2].tolist()
    stat, pval = Contingency.pearsonBlock(x_lst, y_lst, z_lst, pVal=True)
    return stat, pval



####################################

if __name__ == '__main__':

    if 0:  # do selection by date
        USAGE = sys.argv[0] + " dt1 dt2 infile outfile"
        if len(sys.argv) < 5:
            sys.exit(" usage: " + USAGE)
        dt1 = datetime.datetime.strptime(sys.argv[1], "%Y%m%d")
        dt2 = datetime.datetime.strptime(sys.argv[2], "%Y%m%d")
        with open(sys.argv[3], 'r') as infile, open(sys.argv[4], 'w') as outfile:
            select_by_date('row', dt1, dt2, infile, outfile)

    if 1:  # parse questions, answers, users
        USAGE = sys.argv[0] + " postfile userfile"
        if len(sys.argv) < 3:
            sys.exit(" usage: " + USAGE)

        with open(sys.argv[1], 'r') as infile:
            quest_df, ans_df = parse_posts(infile)
            print quest_df.head(), "\n", quest_df.dtypes
            print ans_df.head(), "\n", ans_df.dtypes

        if 0:  # save selected user file
            user_ids = set(quest_df['OwnerUserId'].tolist() + quest_df['OwnerUserId'].tolist())
            with open(sys.argv[2], 'r') as infile, open(sys.argv[3], 'w') as outfile:
                select_by_value('row', 'Id', user_ids, infile, outfile)

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

        # key = 2354299
        # print "quest:\n", quest_df.loc[quest_df['Id'] == key]
        # print "quest:\n", quest_df.loc[key]
        # print "ans:\n", ans_df.loc[ans_df['ParentId'] == key]
        # print "ans_aggreg:\n", answer_aggregs_df.loc[key]

        logging.info("joining answer cols to questions")
        quest_df = quest_df.join(answer_aggregs_df, rsuffix='_answer')
        # print "quest joined:\n", quest_df.loc[quest_df['Id'] == key]
        # print "quest joined:\n", quest_df.loc[key]
        # print quest_df.head(), "\n", quest_df.dtypes

        # print "\n**************************\n"

        # for some reason these is object
        quest_df['OwnerUserId'] = pd.to_numeric(quest_df['OwnerUserId'], downcast='integer')
        user_df['Age'] = pd.to_numeric(user_df['Age'], downcast='integer')


        # print quest_df.head(), "\n", quest_df.dtypes

        # key_user = 230814
        # print "quest:\n", quest_df.loc[quest_df['OwnerUserId'] == key_user]
        # print "user:\n", user_df.loc[user_df['Id'] == key_user]
        # print "user:\n", user_df.loc[key_user]

        logging.info("joining users and questions")
        user_question_df = quest_df.join(user_df, on='OwnerUserId', rsuffix='_user')
        logging.debug("joined table has {} rows".format(len(user_question_df)))
        # print user_question_df.head(), "\n", user_question_df.dtypes

        logging.info("joining users and answers")
        user_answer_df = ans_df.join(user_df, on='OwnerUserId', rsuffix='_user')
        logging.debug("joined table has {} rows".format(len(user_answer_df)))

        # print "\n**************************\n"
        logging.info("joining answers and questions")
        # print "ANSWERS\n", ans_df.head(), "\n", ans_df.dtypes
        # print "QUESTIONS\n", quest_df.head(), "\n", quest_df.dtypes
        # key = 2353228
        # print "ans:\n", ans_df.loc[ans_df['ParentId'] == key]
        # print "quest:\n", quest_df.loc[quest_df['Id'] == key]
        # print "quest:\n", quest_df.loc[key]
        question_answer_df = ans_df.join(quest_df, on='ParentId', rsuffix='_quest')
        # print "question_answer:\n", question_answer_df.loc[question_answer_df['ParentId'] == key]
        logging.debug("joined table has {} rows".format(len(question_answer_df)))
        # print "JOINED\n", question_answer_df.head(300), "\n", question_answer_df.dtypes


        with open("indy_results.tsv", 'w') as resultfile:

            # xy_attrs = QUESTION_FIELDS + ['Length'] + [ v.items()[0][0] for v in aggs.values() ]
            xy_attrs = ['Score', 'CommentCount', 'ViewCount', 'AnswerCount', 'FavoriteCount',
                        'AcceptedAnswerId', 'Length',
                        'mean_CommCount', 'mean_Length', 'mean_Score']
            logging.info("x, y attrs: {}".format(xy_attrs))

            z_attrs_disc = ['Id_user', 'Location']
            z_attrs_cont = ['Reputation', 'Views', 'UpVotes', 'DownVotes', 'Age']

            for i, x in enumerate(xy_attrs):
                for y in xy_attrs[i+1:]:
                    # xy_mat = user_question_df.as_matrix(columns=[x, y])
                    # x_arr = xy_mat[:, 0]
                    # y_arr = xy_mat[:, 1]
                    # x_lst = x_arr.tolist()
                    # y_lst = y_arr.tolist()
                    logging.debug("examining x={}, y={}".format(x, y))

                    # marg = Contingency.rsquare(xvals, yvals)
                    # r, pval = Contingency.stats.pearsonr(x_arr, y_arr)
                    stat, pval = pearson(user_question_df, x, y)
                    logging.debug("marg rsquare: {} p={}".format(stat, pval))
                    resultfile.write("q.{}\tq.{}\t{}\t{}\t{}\n".format(x, y, None, stat, pval))

                    for z in z_attrs_disc:
                        # z_lst = user_question_df[z].tolist()
                        # stat, pval = Contingency.pearsonBlock(x_lst, y_lst, z_lst, pVal=True)
                        stat, pval = pearson_block(user_question_df, x, y, z)
                        logging.debug("cond {}: {} {}".format(z, stat, pval))
                        resultfile.write("q.{}\tq.{}\tu.{}\t{}\t{}\n".format(x, y, z, stat, pval))

                    for z in z_attrs_cont:
                        # zvals = user_question_df.as_matrix(columns=[z])
                        # stat, pval = Contingency.partial_corr(x_arr, y_arr, zvals, pval=True)
                        stat, pval = pearson_partial(user_question_df, x, y, z)
                        logging.debug("cond {}: {} {}".format(z, stat, pval))
                        resultfile.write("q.{}\tq.{}\tu.{}\t{}\t{}\n".format(x, y, z, stat, pval))

                    logging.debug("\n")


            xy_attrs = ['Score', 'CommentCount', 'Length'] #, 'Accepted']
            for i, x in enumerate(xy_attrs):
                for y in xy_attrs[i+1:]:

                    # xy_mat = user_answer_df.as_matrix(columns=[x, y])
                    # x_arr = xy_mat[:, 0]
                    # y_arr = xy_mat[:, 1]
                    # x_lst = x_arr.tolist()
                    # y_lst = y_arr.tolist()
                    logging.debug("examining x={}, y={}".format(x, y))

                    # marg = Contingency.rsquare(xvals, yvals)
                    # r, pval = Contingency.stats.pearsonr(x_arr, y_arr)
                    stat, pval = pearson(user_question_df, x, y)
                    logging.debug("marg rsquare: {} p={}".format(stat, pval))
                    resultfile.write("q.{}\tq.{}\t{}\t{}\t{}\n".format(x, y, None, stat, pval))

                    for z in z_attrs_disc:
                        # z_lst = user_answer_df[z].tolist()
                        # stat, pval = Contingency.pearsonBlock(x_lst, y_lst, z_lst, pVal=True)
                        stat, pval = pearson_block(user_question_df, x, y, z)
                        logging.debug("cond {}: {} {}".format(z, stat, pval))
                        resultfile.write("q.{}\tq.{}\tu.{}\t{}\t{}\n".format(x, y, z, stat, pval))

                    for z in z_attrs_cont:
                        # zvals = user_answer_df.as_matrix(columns=[z])
                        # stat, pval = Contingency.partial_corr(x_arr, y_arr, zvals, pval=True)
                        stat, pval = pearson_partial(user_question_df, x, y, z)
                        logging.debug("cond {}: {} {}".format(z, stat, pval))
                        resultfile.write("q.{}\tq.{}\tu.{}\t{}\t{}\n".format(x, y, z, stat, pval))

                    logging.debug("\n")



            xy_attrs = ['Score', 'CommentCount', 'Length'] #, 'Accepted']
            z_attrs_disc = ['Id_quest']
            z_attrs_cont = ['Score_quest', 'CommentCount_quest', 'ViewCount', 'AnswerCount',
                            'FavoriteCount', 'Length_quest']
            for i, x in enumerate(xy_attrs):
                for y in xy_attrs[i+1:]:
                    logging.debug("examining x={}, y={}".format(x, y))

                    stat, pval = pearson(question_answer_df, x, y)
                    logging.debug("marg rsquare: {} p={}".format(stat, pval))
                    resultfile.write("a.{}\ta.{}\t{}\t{}\t{}\n".format(x, y, None, stat, pval))

                    for z in z_attrs_disc:
                        stat, pval = pearson_block(question_answer_df, x, y, z)
                        logging.debug("cond {}: {} {}".format(z, stat, pval))
                        resultfile.write("a.{}\ta.{}\tq.{}\t{}\t{}\n".format(x, y, z, stat, pval))

                    for z in z_attrs_cont:
                        stat, pval = pearson_partial(question_answer_df, x, y, z)
                        logging.debug("cond {}: {} {}".format(z, stat, pval))
                        resultfile.write("a.{}\ta.{}\tq.{}\t{}\t{}\n".format(x, y, z, stat, pval))













