# importing the libraries
import urllib.request
from bs4 import BeautifulSoup
import requests
import warnings
warnings.filterwarnings("ignore")

url="https://en.wikipedia.org/wiki/Google"

html = urllib.request.urlopen(url).read()
soup = BeautifulSoup(html,"html.parser")

for script in soup(["script", "style"]):
    script.decompose()

strips = list(soup.stripped_strings)
# print(strips[:5])

file = open('input.txt','w',encoding="utf-8")

for line in strips:
    file.write(str(line))

print(file.name)
file.close()