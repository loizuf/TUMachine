import csv

def write_subms(dataset_test, predictions, method_name):
    subm = "/home/felentovic/Documents/TUWien/Semester_3/Machine_Learning/Excercise1/cup98ID.predictions_"+method_name+".csv"
    file_subm = open(subm, "w")
    pamwriter = csv.writer(file_subm)

    ids = dataset_test["CONTROLN"]
    pamwriter.writerow(["CONTROLN", "TARGET_B"])
    counter = 0
    for index in range(len(predictions)):
        if predictions[index] == 1:
            counter+=1

        # print([ids[index],predictions[index]])
        pamwriter.writerow([ids[index], predictions[index]])
    print("predicted 1s",counter)
    file_subm.close()