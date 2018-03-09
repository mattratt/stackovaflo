import sys
import logging
import datetime
import xml.etree.ElementTree as ET


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


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


####################################

if __name__ == '__main__':

    if 1:  # do selection by date
        USAGE = sys.argv[0] + " dt1 dt2 infile outfile"
        dt1 = datetime.datetime.strptime(sys.argv[1], "%Y%m%d")
        dt2 = datetime.datetime.strptime(sys.argv[2], "%Y%m%d")
        with open(sys.argv[3], 'r') as infile, open(sys.argv[4], 'w') as outfile:
            select_by_date('row', dt1, dt2, infile, outfile)


