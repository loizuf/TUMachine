malicious_data_train_file = open("G:/OrderForLinux/Universitaet/Wien/3.Semester/Machine_Learning/Virus/Dataset/dataset.train", "r")
cleaned_data = open("G:/OrderForLinux/Universitaet/Wien/3.Semester/Machine_Learning/Virus/Dataset/dataset_cleaned.train", "w")
cleaned_data_min = open("G:/OrderForLinux/Universitaet/Wien/3.Semester/Machine_Learning/Virus/Dataset/dataset_cleaned_minimal.csv", "w")

data_lines = malicious_data_train_file.readlines()

error_count = 0

header = ""
for x in range(1,514):
    header += str(x) + ","
header += "class\n"
cleaned_data_min.write(header)

# for every entry
for line in data_lines:
    next_line_array = line.split()
    classification = next_line_array[0]
    cleaned_line = "" + classification
    minimal_line = ""
    last_index = 1

    # for every attribute
    for i in range(1, len(next_line_array)-1):

        # index of attr: 1/0
        tokens = next_line_array[i].split(":")
        if int(tokens[0]) > 513:
            break;

        # "impute" missing values, They are implicitly 0
        for j in range(last_index, int(tokens[0])):
            cleaned_line += " " + str(j) + ":0"
            minimal_line += "0,"

        last_index = int(tokens[0]) + 1
        cleaned_line += " " + next_line_array[i]
        minimal_line += "1,"

    # add missing last attributes
    for k in range(last_index, 514):
        cleaned_line += " " + str(k) + ":0"
        minimal_line += "0,"

    cleaned_line += " -1\n"
    minimal_line += classification + "\n"
    cleaned_data.write(cleaned_line)
    cleaned_data_min.write(minimal_line)

print(error_count)
cleaned_data.close()
malicious_data_train_file.close()
