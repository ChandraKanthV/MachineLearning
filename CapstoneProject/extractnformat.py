#!/usr/bin/python

import os
import zipfile
zip_ref = zipfile.ZipFile('./roam_prescription_based_prediction.jsonl.zip', 'r')
zip_ref.extractall('.')
zip_ref.close()


filename = "./roam_prescription_based_prediction.jsonl"

wf = open('roam_predcription_based_predicition_c.jsonl', 'w')

with open(filename) as f:
    wf.write("[")  
    for line in f:
        line = line.replace('\n',',\n')
        wf.write(line)  
    
    wf.seek(-2, os.SEEK_END)
    wf.write("]")  

wf.close()
f.close()

os.remove('./roam_prescription_based_prediction.jsonl')
