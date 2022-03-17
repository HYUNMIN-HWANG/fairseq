import csv



with open("manifest/train.tsv") as f:
    train_data = f.read()
    copy_train = "/output/train.tsv"
    copy_train.write(train_data)

    tr = csv.reader(f, delimiter='\t')
    for row in tr:
        print(row)

with open("manifest/valid.tsv") as f:
    valid_data = f.read()
    copy_valid = "/output/valid.tsv"
    copy_valid.write(valid_data)

    tr = csv.reader(f, delimiter='\t')
    for row in tr:
    	print(row)


