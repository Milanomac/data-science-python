import requests
from bs4 import BeautifulSoup
import pandas as pd

# Original website: https://nationalbanken.statistikbank.dk/nbf/99541
# The html code above contained a frameset. It is used to inform the browser of the division of the screen into different split windows, 
# and prohibits any content inside the body associated with a page. I found the link below within the frameset

page = requests.get("https://nationalbanken.statistikbank.dk/statbank5a/selectvarval/saveselections.asp?MainTable=DNRENTD&PLanguage=0&TableStyle=&Buttons=5&PXSId=99541&IQY=&TC=&ST=ST&rvar0=&rvar1=&rvar2=&rvar3=&rvar4=&rvar5=&rvar6=&rvar7=&rvar8=&rvar9=&rvar10=&rvar11=&rvar12=&rvar13=&rvar14=")
soup = BeautifulSoup(page.content, 'html.parser')
# print(soup.prettify())

table = soup.find(id="pxtable")
# print(type(table))

all_items = table.find_all('tr')
# all_items

period_names = [pn.get_text() for pn in table.select(".pxtable .headfirst")]
# print(period_names)

list = []
for x in table.find_all('tr'):
    nationalbankes_r = [na.get_text() for na in x.select(".No")]
    list.append(nationalbankes_r)

df = pd.DataFrame.from_dict({
    'Date': period_names[1:],
    'Nationalbankens rente - Diskonto (Aug. 1987- )': list[4],
    'Nationalbankens rente - Folioindskud (Aug 1987- )': list[5],
    'Nationalbankens rente - Udlån (Apr. 1992-)': list[6],
    'Nationalbankens rente - Indskudsbeviser (Apr. 1992-)': list[7]
}).set_index('Date')

df = df.T

# Writing dataframe to csv
df.to_csv("interest_rates_DK.csv")