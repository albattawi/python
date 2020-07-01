# import the library used to query a website
import urllib.request
# import the Beautiful soup functions to parse the data returned from the website
from bs4 import BeautifulSoup
# import regex to parse and find info
import re

# define varible to save the url
wiki = 'https://en.wikipedia.org/wiki/Deep_learning'

#Query the website and return the html to the variable 'page'
page = urllib.request.urlopen(wiki)

#Parse the html in the 'page' variable, and store it in Beautiful Soup format
soup = BeautifulSoup(page,"html.parser")
#Beautiful Soup allows you to find that specific element easily by its ID:
print(soup.title.string)

# Open text file to write output search results to it
file = open('wiki_links.txt','w')
file.write(soup.title.string)
file.write('\n================================================================\n')

for line in soup.find_all('a    '):#Searching all the links in html
    formatLink=(str(line.get('href')).strip()) #format a link
    if '/' in formatLink: #
        if '/wiki/' in formatLink:
            file.write("https://en.wikipedia.org/" + formatLink+'\n')
        else:
            file.write(formatLink+'\n')

# read soup html line by line and search all links
#for line in soup.find_all('a'):
#    AllLinks = str(line.get('href')).strip()
#    if '/' in line:
#        if '/wiki/' in line:
#            file.write("https://en.wikipedia.org/" + AllLinks)
#        elif '/' in line:
#            file.write(AllLinks)

print(file.name)
file.close()


# Use function “prettify” to look at nested structure of HTML pag
    #oup.find(id='ResultsContainer')
    #pagehtml = soup.prettify()

