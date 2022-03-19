# Scrapping EBanglaLibrary
from ebanglalibrary_scrape import runscrape
import sys

if __name__ == '__main__':
    if len(sys.argv) == 3:
        try:
            begin = int(sys.argv[1])
            maxid = int(sys.argv[2])
            runscrape(begin=begin, maxid=maxid)
        except Exception as e:
            print(e)