#!/usr/bin/env python3
from bs4 import BeautifulSoup as bs
import xml.etree.ElementTree as etree
import requests
from threading import Thread
from zipfile import ZipFile
from io import BytesIO

def download(link):
    print("Downloading '{}'.".format(link))
    req = requests.get(link)#,stream=True)
    if req.status_code == 200:
        f = ZipFile(BytesIO(req.content))#req.raw
        print('boop')
        f.close()
    else:
        req.raise_for_status()

def download_all(links):
    threads = []
    for link in links:
        t = Thread(target=download,args=(link,),name='dl')
        threads.append(t)
        t.start()
    print('Waiting for downloads to finish ...')
    for t in threads:
        t.join()
    print('Downloads finished.')

def get_year(year):
    urlbase = "https://data.uspto.gov/data2/patent/grant/redbook/fulltext/{}"
    url = urlbase.format(year)
    print('Requesting listing for year year {}'.format(year))
    req = requests.get(url)
    if req.status_code == 200:
        page = bs(req.content,'html.parser')
        links = page.find_all('a')
        dl_links = filter(lambda l: l['href'].endswith('.zip'), links)
        dl_links = (urlbase.format('{}/{}'.format(year,l.string)) for l in dl_links)
        download_all(dl_links)
    else:
        req.raise_for_status()


# main
min_year = 1976 # available: 1976
max_year = 1977 # available: 2015
for year in range(min_year, max_year):
    get_year(year)
