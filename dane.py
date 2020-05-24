import csv
import numpy as np
from NeuralNetwork import NeuralNetwork


def convert_data(ind, headline, course):
    r = ind
    for row in csvreader:
        colShift = 0
        for c in range(len(headline)):
            if headline[c] == 'school':
                if row[c] == 'GP':
                    data[r, c] = 1
                else:
                    data[r, c] = 0

            if headline[c] == 'sex':
                if row[c] == 'F':
                    data[r, c] = 1
                else:
                    data[r, c] = 0

            if headline[c] == 'address':
                if row[c] == 'U':
                    data[r, c] = 1
                else:
                    data[r, c] = 0

            if headline[c] == 'famsize':
                if row[c] == 'GT3':
                    data[r, c] = 1
                else:
                    data[r, c] = 0

            if headline[c] == 'Pstatus':
                if row[c] == 'T':
                    data[r, c] = 1
                else:
                    data[r, c] = 0

            if headline[c] == 'Mjob':
                if row[c] == 'teacher':
                    data[r, c] = 1
                elif row[c] == 'health':
                    data[r, c+1] = 1
                elif row[c] == 'services':
                    data[r, c+2] = 1
                elif row[c] == 'at_home':
                    data[r, c+3] = 1
                elif row[c] == 'other':
                    data[r, c+4] = 1
                colShift = colShift + 4

            if headline[c] == 'Fjob':
                if row[c] == 'teacher':
                    data[r, c+colShift] = 1
                elif row[c] == 'health':
                    data[r, c+colShift+1] = 1
                elif row[c] == 'services':
                    data[r, c+colShift+2] = 1
                elif row[c] == 'at_home':
                    data[r, c+colShift+3] = 1
                elif row[c] == 'other':
                    data[r, c+colShift+4] = 1
                colShift = colShift + 4

            if headline[c] == 'reason':
                if row[c] == 'home':
                    data[r, c+colShift] = 1
                elif row[c] == 'reputation':
                    data[r, c+colShift+1] = 1
                elif row[c] == 'course':
                    data[r, c+colShift+2] = 1
                elif row[c] == 'other':
                    data[r, c+colShift+3] = 1
                colShift = colShift + 3

            if headline[c] == 'guardian':
                if row[c] == 'mother':
                    data[r, c+colShift] = 1
                elif row[c] == 'father':
                    data[r, c+colShift+1] = 1
                elif row[c] == 'other':
                    data[r, c+colShift+2] = 1
                colShift = colShift + 2

            if headline[c] == 'schoolsup':
                if row[c] == 'yes':
                    data[r, c+colShift] = 1
                else:
                    data[r, c+colShift] = 0

            if headline[c] == 'famsup':
                if row[c] == 'yes':
                    data[r, c+colShift] = 1
                else:
                    data[r, c+colShift] = 0

            if headline[c] == 'paid':
                if row[c] == 'yes':
                    data[r, c+colShift] = 1
                else:
                    data[r, c+colShift] = 0

            if headline[c] == 'activities':
                if row[c] == 'yes':
                    data[r, c+colShift] = 1
                else:
                    data[r, c+colShift] = 0

            if headline[c] == 'nursery':
                if row[c] == 'yes':
                    data[r, c+colShift] = 1
                else:
                    data[r, c+colShift] = 0

            if headline[c] == 'higher':
                if row[c] == 'yes':
                    data[r, c+colShift] = 1
                else:
                    data[r, c+colShift] = 0

            if headline[c] == 'internet':
                if row[c] == 'yes':
                    data[r, c+colShift] = 1
                else:
                    data[r, c+colShift] = 0

            if headline[c] == 'romantic':
                if row[c] == 'yes':
                    data[r, c+colShift] = 1
                else:
                    data[r, c+colShift] = 0

            if headline[c] == 'age' or headline[c] == 'Medu' or headline[c] == 'Fedu' or headline[c] == 'traveltime' or \
                    headline[c] == 'studytime' or headline[c] == 'failures' or headline[c] == 'famrel' or headline[
                c] == 'freetime' or headline[c] == 'goout' or headline[c] == 'Dalc' or headline[c] == 'Walc' or \
                    headline[c] == 'health' or headline[c] == 'absences' or headline[c] == 'G1' or headline[
                c] == 'G2' or headline[c] == 'G3':
                data[r, c+colShift] = row[c]

            if headline[c] == 'course':
                data[r, c+colShift] = course
        r = r + 1

    return r


with open('student-mat.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    headlineMat = next(csvreader)
    rowNumberMat = len(csvfile.readlines())
    headlineMat.insert(len(headlineMat), "course")

with open('student-por.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    headlinePor = next(csvreader)
    rowNumberPor = len(csvfile.readlines())
    headlinePor.insert(len(headlinePor), "course")

headline = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob_teacher', 'Mjob_health', 'Mjob_services', 'Mjob_at_home', 'Mjob_other', 'Fjob_teacher', 'Fjob_health', 'Fjob_services', 'Fjob_at_home', 'Fjob_other', 'reason_home', 'reason_reputation', 'reason_course', 'reason_other', 'guardian_mother', 'guardian_father', 'guardian_other', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3', 'course']
data = np.empty([rowNumberMat+rowNumberPor, len(headline)])
rowIndex = 0

with open('student-mat.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    next(csvreader)

    rowIndex = convert_data(rowIndex, headlineMat, 0)

with open('student-por.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    next(csvreader)

    rowIndex = convert_data(rowIndex, headlinePor, 1)

for a in range(rowIndex):
    print(data[a,:])

nn_desired_output = -1 * np.ones([data.shape[0], 5])
for i in range(data.shape[0]):
    nn_desired_output[i, int(data[i, [x == "Dalc" for x in headline]][0] - 1)] = 1
    print(nn_desired_output[i, :])

data = data - np.ones(data.shape) * np.average(data, axis=0)
data = data / (np.ones(data.shape) * np.std(data, axis=0))
nn_input = data[:, [not x.endswith("alc") for x in headline]]

nn_dalc = NeuralNetwork(nn_input.shape[1], np.asarray([20, 20, 5]))

for epoch in range(1000):

    for i in np.random.permutation(np.asarray(tuple(range(nn_input.shape[0])))):
        nn_dalc.propagate(nn_input[i, :], nn_desired_output[i, :])

    errors = np.ndarray([nn_input.shape[0]])
    results = -1 * np.ones([nn_input.shape[0], nn_desired_output.shape[1]])
    error_matrix = np.zeros([5, 6])
    for i in range(nn_input.shape[0]):
        output = nn_dalc.run(nn_input[i, :])
        error_matrix[np.argmax(nn_desired_output[i, :]), np.argmax(output)] += 1
        results[i, np.argmax(output)] = 1
        errors[i] = not np.array_equal(results[i, :], nn_desired_output[i, :])

    print(np.average(errors))
    print(error_matrix)
    np.savetxt("result_"+str(epoch), results)
    np.savetxt("error_matrix_"+str(epoch), error_matrix)
    print(np.average(results, axis=0))

