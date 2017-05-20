from scipy.io import arff
from io import StringIO

import csv

import os

content = os.path.abspath('/home/nikhil/major2/cocomonasa_v1.arff')
data, meta = arff.loadarff(content)
##print(data)

with open('CSV_Version','w') as out:
    csv_out=csv.writer(out)
    for row in data:
        csv_out.writerow(row)

##use meta as the attributes list
