# Data Collection

[ebanglalibrary.com](https://www.ebanglalibrary.com/) is free bangla book reading website. The dataset is made from scraping 104459 pages from
[ebanglalibrary.com](https://www.ebanglalibrary.com/).


## Necessary steps to setup the project
1. Create a virtual environment: `python3 -m venv venv`.
2. Install required packages: `pip3 install -r requirements.txt`
3. Activate the virtual environment: `source venv/bin/activate` [For Linux]
4. Activate the virtual environment: `.\venv\Scripts\activate` [For Windows]

## CLI Instructions

1. `python3 main.py`
2. `python3 main.py <begin_id> <max_id>`
3. `begin_id`: ID where the script start the scrapping
4. `max_id`: ID where the script finish the scrapping
5. For step 1 the script start to scrape from 1 to 104459
6. If a file named `track.txt` found in data directory with a number then begin_id will be ignored
7. To merge different directories scraped using different machines, use `merge-file.py`: `python merge-file.py <dir1> <dir2>`.
8. `merge-file.py` appends files from `dir2` to files of same author name and book name in `dir1`.
