from bs4 import BeautifulSoup
import requests as req
import time
import sys
import os

def scrape(url: str) -> str:
    src = req.get(url)
    soup = BeautifulSoup(src.content, 'html.parser')

    text = soup.find('div', class_='entry-content entry-content-single')

    if text is None:
        return ''

    return text.text


def program_close():
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)


def save_state(track):
    print('\nSaving state....')
    with open('data/track.txt', 'w+') as f:
        f.write(str(track))


def runlinks(base='https://www.ebanglalibrary.com/', maxlink=104453):
    begin = 1
    with open('data/track.txt') as f:
        r = f.read()
        if len(r) != 0:
            try:
                begin = int(r)
            except ValueError:
                pass

    track = begin
    try:
        d_sum = 0
        c = 1
        with open('data/data.txt', 'a') as f:
            for i in range(begin, maxlink + 1):
                start = time.time()
                
                print('Running for link', base+str(i), '...', end='')
                text = scrape(base + str(i))
                f.write(text+'\r')
                track = i + 1

                end = time.time()

                # Estimated Time Calculation
                duration = end-start
                d_sum += duration
                avg_d = d_sum / c
                total_time = avg_d*(maxlink-begin+1)
                elapsed_t = c*avg_d
                time_needed = total_time-elapsed_t
                hours = int(time_needed / 3600)
                extra = time_needed % 3600
                minutes = int(extra / 60)
                seconds = int(extra % 60)

                print(f"Estimated time needed: {hours}:{minutes}:{seconds}.", end='\r')

                c += 1

    except KeyboardInterrupt as ke:
        save_state(track)
        program_close()
    
    except Exception as e:
        save_state(track)
        program_close()