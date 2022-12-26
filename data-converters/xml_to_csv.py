# DESCRIPTION:      This script converts xml file to csv using xml.etree and pandas. 
#                   The converted data are official NBP interest rates.

from xml.etree import ElementTree as ET
import pandas as pd
import urllib

import urllib.request 
urllib.request.urlretrieve("https://www.nbp.pl/xml/stopy_procentowe.xml", "int_rates.xml")

COLS = ["id", "nazwa", "name", "oprocentowanie", "obowiazuje_od"]

tree = ET.parse("int_rates.xml")
rows = []

for i in tree.findall("tabela/pozycja"):

# Defining attributes, help here: https://stackoverflow.com/questions/70634926/problem-with-converting-xml-file-to-csv-file-in-python
    id = i.attrib.get("id")
    nazwa = i.attrib.get("nazwa")
    name = i.attrib.get("name")
    oprocentowanie = i.attrib.get("oprocentowanie")
    obowiazuje_od = i.attrib.get("obowiazuje_od")
            
    rows.append({
                "id": id,
                "nazwa": nazwa,
                "Name": name,
                "name": name,
                "oprocentowanie": oprocentowanie,
                "obowiazuje_od": obowiazuje_od  
                })
  
df = pd.DataFrame(rows, columns=COLS)
print(df)
  
# Writing dataframe to csv
df.to_csv("interest_rates_PL.csv", index = False)