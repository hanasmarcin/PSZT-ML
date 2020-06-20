# PSZT-ML Neural Network project
# Hanas Marcin, Tuzimek Radosław
# PW
import csv
import numpy as np
from NeuralNetwork import NeuralNetwork


np.random.seed(53645)
np.set_printoptions(precision=2, floatmode='maxprec_equal')


def convert_data(ind, headline, course):
    r = ind
    for row in csvreader:
        colShift = 0
        for c in range(len(headline)):
            if headline[c] == 'school':
                if row[c] == 'GP':
                    data[r, c] = 1

            elif headline[c] == 'sex':
                if row[c] == 'F':
                    data[r, c] = 1
            elif headline[c] == 'address':
                if row[c] == 'U':
                    data[r, c] = 1

            elif headline[c] == 'famsize':
                if row[c] == 'GT3':
                    data[r, c] = 1

            elif headline[c] == 'Pstatus':
                if row[c] == 'T':
                    data[r, c] = 1

            elif headline[c] == 'Mjob':
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

            elif headline[c] == 'Fjob':
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

            elif headline[c] == 'reason':
                if row[c] == 'home':
                    data[r, c+colShift] = 1
                elif row[c] == 'reputation':
                    data[r, c+colShift+1] = 1
                elif row[c] == 'course':
                    data[r, c+colShift+2] = 1
                elif row[c] == 'other':
                    data[r, c+colShift+3] = 1
                colShift = colShift + 3

            elif headline[c] == 'guardian':
                if row[c] == 'mother':
                    data[r, c+colShift] = 1
                elif row[c] == 'father':
                    data[r, c+colShift+1] = 1
                elif row[c] == 'other':
                    data[r, c+colShift+2] = 1
                colShift = colShift + 2

            elif headline[c] == 'schoolsup':
                if row[c] == 'yes':
                    data[r, c+colShift] = 1

            elif headline[c] == 'famsup':
                if row[c] == 'yes':
                    data[r, c+colShift] = 1

            elif headline[c] == 'paid':
                if row[c] == 'yes':
                    data[r, c+colShift] = 1

            elif headline[c] == 'activities':
                if row[c] == 'yes':
                    data[r, c+colShift] = 1

            elif headline[c] == 'nursery':
                if row[c] == 'yes':
                    data[r, c+colShift] = 1

            elif headline[c] == 'higher':
                if row[c] == 'yes':
                    data[r, c+colShift] = 1

            elif headline[c] == 'internet':
                if row[c] == 'yes':
                    data[r, c+colShift] = 1

            elif headline[c] == 'romantic':
                if row[c] == 'yes':
                    data[r, c+colShift] = 1

            elif headline[c] == 'age' or headline[c] == 'Medu' or headline[c] == 'Fedu' or headline[c] == 'traveltime' or \
                    headline[c] == 'studytime' or headline[c] == 'failures' or headline[c] == 'famrel' or headline[
                c] == 'freetime' or headline[c] == 'goout' or headline[c] == 'Dalc' or headline[c] == 'Walc' or \
                    headline[c] == 'health' or headline[c] == 'absences' or headline[c] == 'G1' or headline[
                c] == 'G2' or headline[c] == 'G3':
                data[r, c+colShift] = row[c]

            elif headline[c] == 'mat':
                if course == 0:
                    data[r, c+colShift] = 1

            elif headline[c] == 'por':
                if course == 1:
                    data[r, c+colShift] = 1

        indices = [np.all(np.equal(data[i, 0:-2], data[r, 0:-2])) and i != r for i in range(data.shape[0])]
        if sum(indices) > 0:
            print(data[indices, -1])
            if course == 1:
                data[indices, -1][0] = 1
            else:
                data[indices, -2][0] = 1
        else:
            r = r + 1
    return r


with open('student-mat.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    headlineMat = next(csvreader)
    rowNumberMat = len(csvfile.readlines())
    headlineMat.insert(len(headlineMat), "mat")
    headlineMat.insert(len(headlineMat), "por")

with open('student-por.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    headlinePor = next(csvreader)
    rowNumberPor = len(csvfile.readlines())
    headlinePor.insert(len(headlinePor), "mat")
    headlinePor.insert(len(headlinePor), "por")

headline = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob_teacher', 'Mjob_health', 'Mjob_services', 'Mjob_at_home', 'Mjob_other', 'Fjob_teacher', 'Fjob_health', 'Fjob_services', 'Fjob_at_home', 'Fjob_other', 'reason_home', 'reason_reputation', 'reason_course', 'reason_other', 'guardian_mother', 'guardian_father', 'guardian_other', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3', 'mat', 'por']
data = np.zeros([rowNumberMat+rowNumberPor, len(headline)])
rowIndex = 0

with open('student-mat.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    next(csvreader)

    rowIndex = convert_data(rowIndex, headlineMat, 0)

with open('student-por.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    next(csvreader)

    rowIndex = convert_data(rowIndex, headlinePor, 1)

data = data[0:rowIndex, :]

# Choose learn and test data randomly from data
print(data.shape)
rand_indices = np.random.permutation(data.shape[0])
train_data = data[list(rand_indices[0:3*data.shape[0]//4]), :]
test_data = data[list(rand_indices[3*data.shape[0]//4:data.shape[0]]), :]

# Create histogram data
alc_count = np.zeros([5, 5, 2])
for i in range(train_data.shape[0]):
    alc_count[int(train_data[i, [x == "Dalc" for x in headline]][0] - 1), int(train_data[i, [x == "Walc" for x in headline]][0] - 1), 0] += 1
for i in range(test_data.shape[0]):
    alc_count[int(test_data[i, [x == "Dalc" for x in headline]][0] - 1), int(test_data[i, [x == "Walc" for x in headline]][0] - 1), 1] += 1

print("Liczba probek dla zbioru uczacego:")
print(alc_count[:, :, 0])
print(np.sum(alc_count[:, :, 0], axis=0))
print(np.sum(alc_count[:, :, 0], axis=1))
print("Liczba probek dla zbioru testujacego:")
print(alc_count[:, :, 1])
print(np.sum(alc_count[:, :, 1], axis=0))
print(np.sum(alc_count[:, :, 1], axis=1))
print("Liczba próbek dla wszystkich danych:")
print(np.sum(alc_count, axis=2))

# Create vectors with desired network outputs for test and train datasets
train_desired_dalc_output = -1 * np.ones([train_data.shape[0], 5])
train_desired_walc_output = -1 * np.ones([train_data.shape[0], 5])
test_desired_dalc_output = -1 * np.ones([test_data.shape[0], 5])
test_desired_walc_output = -1 * np.ones([test_data.shape[0], 5])
for i in range(train_data.shape[0]):
    train_desired_dalc_output[i, int(train_data[i, [x == "Dalc" for x in headline]][0] - 1)] = 1
    train_desired_walc_output[i, int(train_data[i, [x == "Walc" for x in headline]][0] - 1)] = 1
for i in range(test_data.shape[0]):
    test_desired_dalc_output[i, int(test_data[i, [x == "Dalc" for x in headline]][0] - 1)] = 1
    test_desired_walc_output[i, int(test_data[i, [x == "Walc" for x in headline]][0] - 1)] = 1

train_data = train_data[:, [not x.endswith("alc") for x in headline]]
test_data = test_data[:, [not x.endswith("alc") for x in headline]]

# Calculate means and std
train_mean = np.mean(train_data, axis=0)
train_std = np.std(train_data, axis=0)
print(train_mean)
print(train_std)

# Normalize train and test data using mean and std of the train data
train_data = train_data - np.ones(train_data.shape) * train_mean
train_data = train_data / (np.ones(train_data.shape) * train_std)

test_data = test_data - np.ones(test_data.shape) * train_mean
test_data = test_data / (np.ones(test_data.shape) * train_std)

# features_count = 25 # data.shape[1]
# nn_train_input = train_data[:, np.argsort(train_std)[-features_count:]]
# nn_test_input = test_data[:, np.argsort(train_std)[-features_count:]]
nn_train_input = train_data
nn_test_input = test_data
# Choose attributes that will be the input for neural network
# nn_train_input = nn_train_input[:, 0:-20]
# nn_test_input = nn_test_input[:, 0:-20]

# Create neural network for dalc
nn_dalc = NeuralNetwork(nn_train_input.shape[1], np.asarray([30, 5]), 0.025)

for epoch in range(50):
    print(epoch)
    for i in np.random.permutation(np.asarray(tuple(range(nn_train_input.shape[0])))):
        nn_dalc.propagate(nn_train_input[i, :], train_desired_dalc_output[i, :])

    # true, if element was classified incorrectly
    errors_train = np.ndarray([nn_train_input.shape[0]])
    # transformed output of neural network, for each element 1 for choosen class, -1 for the rest
    results_train = -1 * np.ones([nn_train_input.shape[0], train_desired_dalc_output.shape[1]])
    # matrix with count, which class the element really is (row id) and in which was it classified (column id)
    error_matrix_train = np.zeros([5, 5])
    for i in range(nn_train_input.shape[0]):
        output = nn_dalc.run(nn_train_input[i, :])
        error_matrix_train[np.argmax(train_desired_dalc_output[i, :]), np.argmax(output)] += 1
        results_train[i, np.argmax(output)] = 1
        errors_train[i] = not np.array_equal(results_train[i, :], train_desired_dalc_output[i, :])

    # true, if element was classified incorrectly
    errors_test = np.ndarray([nn_test_input.shape[0]])
    # transformed output of neural network, for each element 1 for choosen class, -1 for the rest
    results_test = -1 * np.ones([nn_test_input.shape[0], test_desired_dalc_output.shape[1]])
    # matrix with count, which class the element really is (row id) and in which was it classified (column id)
    error_matrix_test = np.zeros([5, 5])
    for i in range(nn_test_input.shape[0]):
        output = nn_dalc.run(nn_test_input[i, :])
        error_matrix_test[np.argmax(test_desired_dalc_output[i, :]), np.argmax(output)] += 1
        results_test[i, np.argmax(output)] = 1
        errors_test[i] = not np.array_equal(results_test[i, :], test_desired_dalc_output[i, :])

    print(np.average(errors_train))
    print(error_matrix_train)
    print(np.average(errors_test))
    print(error_matrix_test)


# Create neural network for dalc
nn_walc = NeuralNetwork(nn_train_input.shape[1], np.asarray([60, 5]), 0.028)

for epoch in range(50):
    print(epoch)
    for i in np.random.permutation(np.asarray(tuple(range(nn_train_input.shape[0])))):
        nn_walc.propagate(nn_train_input[i, :], train_desired_walc_output[i, :])

    # true, if element was classified incorrectly
    errors_train = np.ndarray([nn_train_input.shape[0]])
    # transformed output of neural network, for each element 1 for choosen class, -1 for the rest
    results_train = -1 * np.ones([nn_train_input.shape[0], train_desired_walc_output.shape[1]])
    # matrix with count, which class the element really is (row id) and in which was it classified (column id)
    error_matrix_train = np.zeros([5, 5])
    for i in range(nn_train_input.shape[0]):
        output = nn_walc.run(nn_train_input[i, :])
        error_matrix_train[np.argmax(train_desired_walc_output[i, :]), np.argmax(output)] += 1
        results_train[i, np.argmax(output)] = 1
        errors_train[i] = not np.array_equal(results_train[i, :], train_desired_walc_output[i, :])

    # true, if element was classified incorrectly
    errors_test = np.ndarray([nn_test_input.shape[0]])
    # transformed output of neural network, for each element 1 for chosen class, -1 for the rest
    results_test = -1 * np.ones([nn_test_input.shape[0], test_desired_walc_output.shape[1]])
    # matrix with count, which class the element really is (row id) and in which was it classified (column id)
    error_matrix_test = np.zeros([5, 5])
    for i in range(nn_test_input.shape[0]):
        output = nn_walc.run(nn_test_input[i, :])
        error_matrix_test[np.argmax(test_desired_walc_output[i, :]), np.argmax(output)] += 1
        results_test[i, np.argmax(output)] = 1
        errors_test[i] = not np.array_equal(results_test[i, :], test_desired_walc_output[i, :])

    print(np.average(errors_train))
    print(np.average(errors_test))
    print(error_matrix_train)
    print(error_matrix_test)
