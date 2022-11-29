# DESCRIPTION:      This script converts JSON file to csv using json and pandas. Data in the csv in a melted format.
#                   Script is generic and can work with any of the Lithuania Statistics JSON files from the API (https://osp.stat.gov.lt/web/guest/rdb-rest)
#                   Link to all json files https://osp-rs.stat.gov.lt/rest_xml/dataflow/

import json
import pandas as pd
import urllib.request 

urllib.request.urlretrieve("https://osp-rs.stat.gov.lt/rest_json/data/S7R007_M2010210_1", "lithuania_gdp.json")

with open("lithuania_gdp.json") as file:  
    data=json.load(file)

#creating the data dataframe
actdata=data['dataSets'][0]["observations"]
dfs=pd.DataFrame([[key,actdata[key]] for key in actdata.keys()])

#splitting columns of the data dataframe and joining
dfsi=pd.DataFrame(dfs[0].str.split(':').tolist(),index=dfs.index)
dfsd=pd.DataFrame(dfs[1].tolist(),index=dfs.index)
dfss=pd.merge(dfsi,dfsd,left_index=True,right_index=True).apply(pd.to_numeric)

#dataframe for reference in for loop
dref=dfss

#dictionary to store metadata dataframes 
ndfs={}

#merging with metadata
for iteration, x in enumerate(data['structure']['dimensions']['observation']):
    
    #storing metadata according to their order
    ndfs[iteration]=pd.DataFrame(x["values"])
    #changing column titles to avoid duplicates
    for iter2,y in enumerate(ndfs[iteration].columns):
        ndfs[iteration].rename(columns={(y):('code'+str(iteration)+str(iter2))},inplace=True)
    #column index
    #ci=str(iteration)+'_x'
    ci=dref.columns[iteration]   
    #merging with metadata and deleting index row
    dfss=pd.merge(dfss, ndfs[iteration],left_on=ci,right_index=True,how='left').drop(ci,axis=1)

#Exporting for reading
dfss.to_csv("dgp.csv", index=False)