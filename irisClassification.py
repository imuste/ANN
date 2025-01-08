import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import textwrap
from time import sleep


"""
sigmoid: performs the sigmoid function 
Parameters: the value on which to compute the function
"""
def sigmoid(x):
        return 1 / (1 + np.exp(-x))


"""
sigmoidDerivative: performs the sigmoid derivative function 
Parameters: the value on which to compute the function
"""
def sigmoidDerivative(x):
        return x * (1 - x)


"""
forwardPropegation: does a forward propagation on a set of features using the 
        provided weights and biases
Parameters: the set of features, the biases and the weights
"""
def forwardPropegation(X, weight1, bias1, weight2, bias2):
        # from the input to hidden layer
        hiddenLayerIn = np.dot(X, weight1) + bias1
        hiddenLayerOut = sigmoid(hiddenLayerIn)

        # from the hidden to output layer
        outputLayerIn = np.dot(hiddenLayerOut, weight2) + bias2
        outputLayerOut = sigmoid(outputLayerIn)

        return hiddenLayerIn, hiddenLayerOut, outputLayerIn, outputLayerOut


"""
backwardPropegation: does a backward propagation on a set of features using the 
        provided weights and biases
Parameters: the set of features, the biases, the weights, and layer outputs
"""
def backwardPropegation(X, Y, hiddenLayerIn, hiddenLayerOut, outputLayerIn, 
        outputLayerOut, weight1, weight2, bias1, bias2, learningRate=0.1):
        
        # number of training samples
        numSamples = X.shape[0]

        # get output layer error
        outputError = outputLayerOut - Y
        outWeightGradient = np.dot(hiddenLayerOut.T, outputError) / numSamples
        outBiasGradient = np.sum(outputError, axis=0, keepdims=True) / numSamples

        # get hidden layer error
        hiddenLayerError = np.dot(outputError, weight2.T)
        hiddenLayerDelta = hiddenLayerError * sigmoidDerivative(hiddenLayerOut)
        hiddenWeightGradient = np.dot(X.T, hiddenLayerDelta) / numSamples
        hiddenBiasGradient = np.sum(hiddenLayerDelta, axis=0, keepdims=True) / numSamples

        # recompute weights and biases
        weight1 -= learningRate * hiddenWeightGradient
        bias1 -= learningRate * hiddenBiasGradient
        weight2 -= learningRate * outWeightGradient
        bias2 -= learningRate * outBiasGradient

        return weight1, bias1, weight2, bias2


"""
trainNetwork: trains the network using the provided number of rounds on the 
        provided feature set
Parameters: the number of rounds, learning rate, training sets, weights, and 
        biases
"""
def trainNetwork(rounds, learningRate, X_train, Y_train, weight1, bias1, 
        weight2, bias2):

        print("Training summary with loss progress:")

        for round in range(rounds):
                # do a forward pass
                hiddenLayerIn, hiddenLayerOut, outLayerIn, outLayerOut = \
                forwardPropegation(X_train, weight1, bias1, weight2, bias2)

                # compute the loss
                loss = np.mean((outLayerOut - Y_train) ** 2)

                # do a backward pass
                weight1, bias1, weight2, bias2 = backwardPropegation(X_train, 
                        Y_train, hiddenLayerIn, hiddenLayerOut, 
                        outLayerIn, outLayerOut, weight1, 
                        weight2, bias1, bias2, learningRate)

                # print the loss
                if round % 100 == 0:
                        print(f"  Round {round}, Loss: {loss:.3f}")
        
        sleep(2)
        return weight1, bias1, weight2, bias2



"""
predict: predicts the set of features using the forward propagation with the 
        provided weights and biases
Parameters: the set of features, the weights, and biases
"""
def predict(X, weight1, bias1, weight2, bias2):
        _, _, _, outputLayerOut = \
                forwardPropegation(X, weight1, bias1, weight2, bias2)

        return np.argmax(outputLayerOut, axis=1)


"""
testANN: tests the ANN using the testing set and the prediction function
Parameters: the testing set, the weights and biases
"""
def testANN(X_test, Y_test, weight1, bias1, weight2, bias2):
        Y_pred = predict(X_test, weight1, bias1, weight2, bias2)
        Y_TestLabels = np.argmax(Y_test, axis=1)
        accuracy = np.mean(Y_pred == Y_TestLabels) * 100
        print(f"\nTest Accuracy is: {accuracy:.2f}% after testing the network\n")
        sleep(2)


"""
getUserSamples: prints the validation set to the user and asks the user to 
        choose a sample to classify
Parameters: the validation set, the weights and biases
"""
def getUserSamples(X_validation, weight1, bias1, weight2, bias2, encoder):

        numToFlower = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}

        print("Now you will be able to get the flowers in the following list classified:")
        for item in X_validation:
                print("  Flower attributes:", item)

        # get the garden queries
        while True:
                print("\nPlease enter flower attributes to classify your flower from the provided list by entering its features (SepalLength, SepalWidth, PetalLength, PetalWidth) or 'quit' to quit:")
                userInput = input()
                if userInput.lower() == 'quit':
                        break
                try:
                        flower = np.array([float(x) for x in userInput.split(",")]).reshape(1, -1)
                        prediction = predict(flower, weight1, bias1, weight2, bias2)
                        flowerClass = encoder.inverse_transform(np.eye(3)[prediction])[0][0]
                        print(f"The flower is a: {numToFlower[flowerClass]}!")
                except ValueError:
                        print("Invalid input. Please enter the comma separated numeric values.")


"""
programInit: prints out the initial message to the user
"""
def programInit():
        print()
        print("================================================================================\n")
        
        print("WELCOME TO THE FLOWER CLASSIFIER")
        print(textwrap.fill("This program will use samples of Iris flower types to create a classifier that can use the flower attributes to classify the type of Iris it is.\n"))
        print(textwrap.fill("The algorithm used to determine the type of Iris is the Artificial Neural Network algorithm.\n"))
        print(textwrap.fill("After the network has been trained and tested, you will get a list of the attributes of 10 flowers which you can enter to get the program to determine what type of Iris the flower is."))
        print()

        print("================================================================================")
        print()
        print()

        sleep(5)
        


"""
main: runs the general ANN by splitting the data set, trainig the network, 
        testing it, and getting user input to classify unseen samples
"""
def main():

        programInit()

        # read in the file and structure the data
        irisData = pd.read_csv("ANN _Iris_Data.txt", header=None, 
                                names=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Class"])
        
        # assign numbers to the flower lables
        irisData['Class'] = irisData['Class'].map({'Iris-setosa': 0, 
                'Iris-versicolor': 1, 'Iris-virginica': 2})
        
        # separate features and labels
        X = irisData.iloc[:, :-1].values
        Y = irisData.iloc[:, -1].values

        # encode labels for multiclass classification
        encoder = OneHotEncoder(sparse_output=False)
        Y = encoder.fit_transform(Y.reshape(-1, 1))

        # split data into training (70%), testing (20%), validation sets (10%)
        X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, 
                stratify=Y, random_state=42)
        X_test, X_validation, Y_test, Y_validation = train_test_split(X_temp, 
                Y_temp, test_size=1/3, stratify=Y_temp, random_state=42)

        # 4 input neurons, 5 hidden neurons, and 3 output neurons
        inputSize = 4
        hiddenSize = 5
        outputSize = 3
        rounds = 1500  # number of training rounds
        learningRate = 0.1

        # assign weights and biases
        weight1 = np.random.randn(inputSize, hiddenSize) * 0.01
        bias1 = np.zeros((1, hiddenSize))
        weight2 = np.random.randn(hiddenSize, outputSize) * 0.01 
        bias2 = np.zeros((1, outputSize))

        # train the ANN
        weight1, bias1, weight2, bias2 = trainNetwork(rounds, learningRate, 
                X_train, Y_train, weight1, bias1, weight2, bias2)

        # test the ANN
        testANN(X_test, Y_test, weight1, bias1, weight2, bias2)

        # get user samples from validation set
        getUserSamples(X_validation, weight1, bias1, weight2, bias2, encoder)
        


main()