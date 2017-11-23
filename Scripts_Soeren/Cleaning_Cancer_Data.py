cancer_train_data = open("G:/OrderForLinux/Universitaet/Wien/3.Semester/Machine_Learning/Cancer/breast-cancer.shuf.train.csv", "r")
complete_train_data_complete = open("G:/OrderForLinux/Universitaet/Wien/3.Semester/Machine_Learning/Cancer/cleaning/breast-cancer.shuf.train.complete.csv", "w")
missing_train_data_caps = open("G:/OrderForLinux/Universitaet/Wien/3.Semester/Machine_Learning/Cancer/cleaning/breast-cancer.shuf.train.missing_caps.csv", "w")

cancer_test_data = open("G:/OrderForLinux/Universitaet/Wien/3.Semester/Machine_Learning/Cancer/breast-cancer.shuf.test.csv", "r")
complete_test_data_complete = open("G:/OrderForLinux/Universitaet/Wien/3.Semester/Machine_Learning/Cancer/cleaning/breast-cancer.shuf.test.complete.csv", "w")
missing_test_data_caps = open("G:/OrderForLinux/Universitaet/Wien/3.Semester/Machine_Learning/Cancer/cleaning/breast-cancer.shuf.test.missing_caps.csv", "w")
missing_test_data_quad = open("G:/OrderForLinux/Universitaet/Wien/3.Semester/Machine_Learning/Cancer/cleaning/breast-cancer.shuf.test.missing_quad.csv", "w")

train_lines = cancer_train_data.readlines()
test_lines = cancer_test_data.readlines()

complete_train_data_complete.write(train_lines[0])
missing_train_data_caps.write(train_lines[0])

complete_test_data_complete.write(train_lines[0])
missing_test_data_caps.write(train_lines[0])
missing_test_data_quad.write(train_lines[0])

for i in range(1, len(train_lines)):
    next_line_array = train_lines[i].split(",")
    if next_line_array[5] == "?":
        missing_train_data_caps.write(train_lines[i])
    else:
        complete_train_data_complete.write(train_lines[i])

for i in range(1, len(test_lines)):
    next_line_array = test_lines[i].split(",")
    if next_line_array[5] == "?":
        missing_test_data_caps.write(test_lines[i])
    elif next_line_array[8] == "?":
        missing_test_data_quad.write(test_lines[i])
    else:
        complete_test_data_complete.write(test_lines[i])

complete_train_data_complete.close()
missing_train_data_caps.close()

complete_test_data_complete.close()
missing_test_data_caps.close()
missing_test_data_quad.close()