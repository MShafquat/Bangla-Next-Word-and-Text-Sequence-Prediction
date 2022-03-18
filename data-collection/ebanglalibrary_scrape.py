from bs4 import BeautifulSoup
import requests as req
import time
import sys
import os
from pathlib import Path


def program_close():
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)


def save_state(id):
    data_dir = Path(__file__).resolve().parent / 'data'
    print('\nSaving state....')
    with open(f'{data_dir}/track.txt', 'w+') as f:
        f.write(str(id))

def scrape(url: str) -> str:
    src = req.get(url)
    soup = BeautifulSoup(src.content, 'html.parser')

    head = soup.findAll('span', {"property": "itemListElement"})
    if len(head) == 3 or len(head) == 4:
        try:
            author = head[1].text.strip()
            book = head[-1].text.strip()
            header = soup.find('span', class_='post post-post current-item').text.strip()
            ptags = soup.find('div', class_='entry-content entry-content-single').findAll('p')
            textp = [p.text for p in ptags]
            text = ' '.join(textp) 
            return author, book, header, text
        except AttributeError as ae:
            return ''
    else:
        return ''

def runlinks(base='https://www.ebanglalibrary.com/', begin=1, maxid=104453):
    id = begin

    data_dir = Path(__file__).resolve().parent / 'data'

    try:
        d_sum = 0
        c = 1

        while True:
            start = time.time()
            print('Running for link', base+str(id), '...', end='')
            l = scrape(base + str(id))
            if(len(l) == 0):
                id += 1
                c += 1
                continue
            author = l[0]; book = l[1]; header = l[2]; text = l[3]
            
            author_path = f"{data_dir}/{author}"
            if not os.path.exists(author_path):
                os.makedirs(author_path)
            
            with open(f"{author_path}/{book}.txt", 'a') as f:
                f.write(header + '\n' + text)
            
            id += 1

            end = time.time()

            # Estimated Time Calculation
            duration = end-start
            d_sum += duration
            avg_d = d_sum / c
            total_time = avg_d*(maxid-begin+1)
            elapsed_t = c*avg_d
            time_needed = total_time-elapsed_t
            hours = int(time_needed / 3600)
            extra = time_needed % 3600
            minutes = int(extra / 60)
            seconds = int(extra % 60)
            
            print(f"Estimated time needed: {hours}:{minutes}:{seconds}.", end='\r')
            c += 1

    except KeyboardInterrupt as ke:
        save_state(id)
        program_close()
    
    except Exception as e:
        print(e)
        save_state(id)
        program_close()


def runscrape():
    data_dir = Path(__file__).resolve().parent / 'data'
    try:
        with open(f'{data_dir}/track.txt') as f:
            r = f.read()
            if len(r) != 0:
                try:
                    runlinks(begin=int(r))
                except ValueError:
                    runlinks(begin=1)
    except FileNotFoundError as e:
        runlinks(begin=1)