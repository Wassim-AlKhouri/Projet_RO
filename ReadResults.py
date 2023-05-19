import csv
### Purpose: Read the results from the csv file and print them to the console ###
if __name__ == '__main__':
    ### Insert the path of the csv file here ###
    File_path = 'results\\result_1_2000_2000_0_1_0_(2, 15)_(0.5, 6)_(0.1, 10).csv'

    ### Read the file and print the results ###
    with open(File_path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        data = []
        for row in reader:
            data.append(row)
        for i in range(len(data[0])):
            print(data[0][i], ':', data[1][i])
