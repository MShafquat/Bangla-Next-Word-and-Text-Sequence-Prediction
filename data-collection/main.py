# Scrapping EBanglaLibrary
from ebanglalibrary_scrape import runscrape
import sys

if __name__ == '__main__':
    if len(sys.argv) == 1:
        # Previous Maximum ID 104453
        # Updated Maximum ID 104459
        runscrape(begin_id=1, maxid=104459)
    elif len(sys.argv) == 3:
        try:
            begin = int(sys.argv[1])
            maxid = int(sys.argv[2])
            runscrape(begin=begin, maxid=maxid)
        except Exception as e:
            print(e)
    else:
        print('Usage: python main.py')
        print('Usage: python main.py <begin_id> <max_id>')
        print('begin_id: ID where the script start the scrapping')
        print('max_id: ID where the script finish the scrapping')
        print('If a file named track.txt found in data directory with a number then begin_id will be ignored')