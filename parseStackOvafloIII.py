import sys
import logging
import datetime
import xml.etree.ElementTree as ET
import pandas as pd

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


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
            questions.append(tuple(vals))

        else:  # answer
            vals = [ rec.attrib.get(attr) for attr in ANSWER_FIELDS ] + [body_len]
            answers.append(tuple(vals))

    logging.info("creating DataFrame for {} questions".format(len(questions)))
    index_vals = get_index_vals(questions, QUESTION_FIELDS, 'Id')
    question_df = pd.DataFrame(questions, index=index_vals, columns=QUESTION_FIELDS+['Length'])

    logging.info("creating DataFrame for {} posts".format(len(answers)))
    index_vals = get_index_vals(answers, ANSWER_FIELDS, 'Id')
    answer_df = pd.DataFrame(answers, index=index_vals, columns=ANSWER_FIELDS+['Length'])

    return question_df, answer_df


def get_index_vals(tups, col_names, index_col_name):
    p = col_names.index(index_col_name)
    return [t[p] for t in tups]


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
    index_vals = get_index_vals(users, USER_FIELDS, 'Id')
    return pd.DataFrame(users, index=index_vals, columns=USER_FIELDS)





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
            print quest_df.head()
            print ans_df.head()

        if 0:  # save selected user file
            user_ids = set(quest_df['OwnerUserId'].tolist() + quest_df['OwnerUserId'].tolist())
            with open(sys.argv[2], 'r') as infile, open(sys.argv[3], 'w') as outfile:
                select_by_value('row', 'Id', user_ids, infile, outfile)

        with open(sys.argv[2], 'r') as infile:
            user_df = parse_users(infile)
            print user_df.head()

        # add answer aggregs to questions, then join user table
        answer_aggregs_df = ans_df.groupby('ParentId').agg({'Score': 'mean',
                                                            'CommentCount': 'mean',
                                                            'Length': 'mean'})
        answer_aggregs_df.columns = ["_".join(x) for x in answer_aggregs_df.columns.ravel()]
        print answer_aggregs_df.head()

        logging.info("joining users and questions")
        user_question_df = quest_df.join(user_df, on='OwnerUserId', rsuffix='user_')
        logging.debug("joined table has {} rows".format(len(user_question_df)))
        print user_question_df.head()
