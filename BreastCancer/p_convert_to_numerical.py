import pandas as pd

complete_test_with_ID = pd.read_csv("../datasets/Cancer/imputed/breast-cancer.shuf.test.imput.csv")
complete_train_with_ID = pd.read_csv("../datasets/Cancer/imputed/breast-cancer.shuf.train.imput.csv")

# drop ID column and convert everything to categorical data, then map numerical data on it
ages = {'20-29': 0, '30-39': 1, '40-49': 2, '50-59': 3, '60-69': 4, '70-79': 5}
meno = {'premeno': 0, 'lt40': 1, 'ge40': 2}
size = {'0-4': 0, '5-9': 1, '10-14': 2, '15-19': 3, '20-24': 4, '25-29': 5, '30-34': 6, '35-39': 7, '40-44': 8, '45-49': 9, '50-54': 10}
inv_nodes = {'0-2': 0, '3-5': 1, '6-8': 2, '9-11': 3, '12-14': 4, '15-17': 5, '24-26':6}
caps = {'no': 0, 'yes': 1}
breast = {'left': 0, 'right': 1}
quad = {'central': 0, 'left_low': 1, 'right_low': 2, 'left_up': 3, 'right_up': 4}
rad = {'no': 0, 'yes': 1}
classes = {'no-recurrence-events': 0, "recurrence-events": 1}

complete_train_with_ID["age"] = complete_train_with_ID["age"].astype('category').map(ages)
complete_train_with_ID["menopause"] = complete_train_with_ID["menopause"].astype('category').map(meno)
complete_train_with_ID["tumor-size"] = complete_train_with_ID["tumor-size"].astype('category').map(size)
complete_train_with_ID["inv-nodes"] = complete_train_with_ID["inv-nodes"].astype('category').map(inv_nodes)
complete_train_with_ID["node-caps"] = complete_train_with_ID["node-caps"].astype('category').map(caps)
complete_train_with_ID["breast"] = complete_train_with_ID["breast"].astype('category').map(breast)
complete_train_with_ID["breast-quad"] = complete_train_with_ID["breast-quad"].astype('category').map(quad)
complete_train_with_ID["irradiat"] = complete_train_with_ID["irradiat"].astype('category').map(rad)
complete_train_with_ID["Class"] = complete_train_with_ID["Class"].astype('category').map(classes)

complete_test_with_ID["age"] = complete_test_with_ID["age"].astype('category').map(ages)
complete_test_with_ID["menopause"] = complete_test_with_ID["menopause"].astype('category').map(meno)
complete_test_with_ID["tumor-size"] = complete_test_with_ID["tumor-size"].astype('category').map(size)
complete_test_with_ID["inv-nodes"] = complete_test_with_ID["inv-nodes"].astype('category').map(inv_nodes)
complete_test_with_ID["node-caps"] = complete_test_with_ID["node-caps"].astype('category').map(caps)
complete_test_with_ID["breast"] = complete_test_with_ID["breast"].astype('category').map(breast)
complete_test_with_ID["breast-quad"] = complete_test_with_ID["breast-quad"].astype('category').map(quad)
complete_test_with_ID["irradiat"] = complete_test_with_ID["irradiat"].astype('category').map(rad)

complete_test_with_ID.to_csv("../datasets/Cancer/imputed/breast-cancer.shuf.test.imput.numerical.csv", index=False)
complete_train_with_ID.to_csv("../datasets/Cancer/imputed/breast-cancer.shuf.train.imput.numerical.csv", index=False)
