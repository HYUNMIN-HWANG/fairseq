import csv



with open("manifest/train.tsv") as f:
    tr = csv.reader(f, delimiter='\t')
    for row in tr:
        print(row)

with open("manifest/valid.tsv") as f:
    tr = csv.reader(f, delimiter='\t')
    for row in tr:
    	print(row)


