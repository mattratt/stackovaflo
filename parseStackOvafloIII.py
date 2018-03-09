import logging
import datetime
import xml.etree.ElementTree as ET


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


def select_by_date(tag, start_dt, end_dt, infile, outfile, date_attr='CreationDate'):
    mark = '<' + tag + ' '
    good_count = 0
    bad_count = 0
    for i, line in enumerate(infile):
        if i % 10000 == 0:
            logging.debug("\t{}".format(i))
        if line.lstrip().startswith(mark):
            parsed = ET.fromstring(line)
            if date_attr in parsed.attrib:
                dt = datetime.datetime.strptime(parsed.attrib[date_attr])
                if (dt < start_dt) or (end_dt <= dt):
                    bad_count += 1
                    continue
                else:
                    good_count += 1

        outfile.write(line)
    logging.debug("kept {} / {} {} records".format(good_count, good_count+bad_count, tag))







