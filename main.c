#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mnist.h"

// 6760 Bytes
#define numInputs 784
#define numHiddenNodes 512
#define numHiddenNodes2 256
#define numHiddenNodes3 32
#define numOutputs 10

#define EPSILON 1e-9

// ReLU Activation Function
double relu(double x) {
    return x > 0 ? x : 0;
}

// Derivative of ReLU Activation Function
double dRelu(double x) {
    return x > 0 ? 1 : 0;
}

double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double dSigmoid(double x) { return x * (1 - x); }

double init_weight(int numInput, int numOutput) {
    double limit = sqrt(6.0 / (numInput + numOutput));
    return ((double)rand() / RAND_MAX) * 2 * limit - limit;
}

void one_hot_encode(int label, double* encoded_label, int num_outputs) {
    for (int i = 0; i < num_outputs; ++i) {
        encoded_label[i] = (i == label) ? 1.0 : 0.0;
    }
}

double cross_entropy_loss(double* actual, double* predicted, int num_outputs) {
    double loss = 0.0;
    for (int i = 0; i < num_outputs; ++i) {
        loss -= actual[i] * log(predicted[i] + EPSILON) + (1 - actual[i]) * log(1 - predicted[i] + EPSILON);
    }
    return loss;
}

int get_predicted_label(double* outputLayer, int num_outputs) {
    int predicted_label = 0;
    double max_value = outputLayer[0];
    for (int i = 1; i < num_outputs; ++i) {
        if (outputLayer[i] > max_value) {
            max_value = outputLayer[i];
            predicted_label = i;
        }
    }
    return predicted_label;
}

int get_actual_label(double* encoded_label, int num_outputs) {
    int label = 0;
    double max_val = encoded_label[0];
    for (int i = 1; i < num_outputs; ++i) {
        if (encoded_label[i] > max_val) {
            max_val = encoded_label[i];
            label = i;
        }
    }
    return label;
}

double calculate_accuracy(double (*training_outputs)[10], double (*predictions)[10], int num_images, int num_outputs) {
    int correct_predictions = 0;
    for (int i = 0; i < num_images; ++i) {
        int predicted_label = get_predicted_label(predictions[i], num_outputs);
        int actual_label = get_actual_label(training_outputs[i], num_outputs);

        if (predicted_label == actual_label) {
            correct_predictions++;
        }
    }

    double accuracy_percentage = ((double)correct_predictions / num_images) * 100.0;
    return accuracy_percentage;
}

int main (void) {

    const double lr = 0.09f;

    double total_loss = 0.0;
    double (*predictions)[numOutputs] = (double (*)[numOutputs])malloc(NUM_IMAGES * sizeof(*predictions));  // Pointer for dynamic allocation
    
    double hiddenLayer[numHiddenNodes];
    double hiddenLayer2[numHiddenNodes2];
    double hiddenLayer3[numHiddenNodes3];
    double outputLayer[numOutputs];
    
    double hiddenLayerBias[numHiddenNodes];
    double hiddenLayer2Bias[numHiddenNodes2];
    double hiddenLayer3Bias[numHiddenNodes3];
    double outputLayerBias[numOutputs];

    double hiddenWeights[numInputs][numHiddenNodes];
    double hiddenWeights2[numHiddenNodes][numHiddenNodes2];
    double hiddenWeights3[numHiddenNodes2][numHiddenNodes3];
    double outputWeights[numHiddenNodes3][numOutputs];

    // Flattened and normalized images
    unsigned char (*images)[IMAGE_SIZE][IMAGE_SIZE] = malloc(sizeof(unsigned char) * NUM_IMAGES * IMAGE_AREA);
    double (*training_inputs)[784] = malloc(sizeof(double) * NUM_IMAGES * 784); 
    // One-hot encoded labels
    double (*training_outputs)[10] = malloc(sizeof(double) * NUM_IMAGES * 10);  
    unsigned char *labels = malloc(sizeof(unsigned char) * NUM_IMAGES);
    
    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numHiddenNodes; j++) {
            hiddenWeights[i][j] = init_weight(numInputs, numHiddenNodes);
        }
    }

    for (int i = 0; i < numHiddenNodes; i++) {
        for (int j = 0; j < numHiddenNodes2; j++) {
            hiddenWeights2[i][j] = init_weight(numHiddenNodes, numHiddenNodes2);
        }
    }

    for (int i = 0; i < numHiddenNodes2; i++) {
        for (int j = 0; j < numHiddenNodes3; j++) {
            hiddenWeights3[i][j] = init_weight(numHiddenNodes2, numHiddenNodes3);
        }
    }

    for (int i = 0; i < numHiddenNodes3; i++) {
        for (int j = 0; j < numOutputs; j++) {
            outputWeights[i][j] = init_weight(numHiddenNodes3, numOutputs);
        }
    }

    for (int i = 0; i < numHiddenNodes; i++) {
        hiddenLayerBias[i] = init_weight(numInputs, numHiddenNodes) / 10.0;
    }

    for (int i = 0; i < numHiddenNodes2; i++) {
        hiddenLayer2Bias[i] = init_weight(numHiddenNodes, numHiddenNodes2) / 10.0;
    }

    for (int i = 0; i < numHiddenNodes3; i++) {
        hiddenLayer3Bias[i] = init_weight(numHiddenNodes2, numHiddenNodes3) / 10.0;
    }

    for (int i = 0; i < numOutputs; i++) {
        outputLayerBias[i] = init_weight(numHiddenNodes3, numOutputs) / 10.0;
    }

    int num_images = NUM_IMAGES;
    load_mnist_dataset("train-images.idx3-ubyte", "train-labels.idx1-ubyte", images, labels, &num_images);

    for (int i = 0; i < num_images; ++i) {
        for (int y = 0; y < IMAGE_SIZE; ++y) {
            for (int x = 0; x < IMAGE_SIZE; ++x) {
                training_inputs[i][y * IMAGE_SIZE + x] = images[i][y][x] / 255.0;
            }
        }
        one_hot_encode(labels[i], training_outputs[i], 10);
    }
   
    int numberOfEpochs = 20;
    // Train the neural network for a number of epochs
    for(int epochs=0; epochs < numberOfEpochs; epochs++) {
        total_loss = 0.0;
        for(int i = 0; i < num_images; i++){

            // Forward pass

            // Compute hidden layer activation
            for (int j=0; j<numHiddenNodes; j++) {
                double activation = hiddenLayerBias[j];
                 for (int k=0; k<numInputs; k++) {
                    activation += training_inputs[i][k] * hiddenWeights[k][j];
                }
                hiddenLayer[j] = relu(activation);
            }           
            // Compute activation of the new hidden layer
            for (int j = 0; j < numHiddenNodes2; j++) {
                double activation = hiddenLayer2Bias[j];
                for (int k = 0; k < numHiddenNodes; k++) {
                    activation += hiddenLayer[k] * hiddenWeights2[k][j];
                }
                hiddenLayer2[j] = relu(activation);
            }
            // Compute activation of the new hidden layer
            for (int j = 0; j < numHiddenNodes3; j++) {
                double activation = hiddenLayer3Bias[j];
                for (int k = 0; k < numHiddenNodes2; k++) {
                    activation += hiddenLayer2[k] * hiddenWeights3[k][j];
                }
                hiddenLayer3[j] = relu(activation);
            }
            // Compute output layer activation
            for (int j = 0; j < numOutputs; j++) {
                double activation = outputLayerBias[j];
                for (int k = 0; k < numHiddenNodes3; k++) {
                    activation += hiddenLayer3[k] * outputWeights[k][j];
                }
                outputLayer[j] = sigmoid(activation);
            }
            // Backprop
            // Compute change in output weights
            double deltaOutput[numOutputs];
            for (int j=0; j<numOutputs; j++) {
                double errorOutput = (training_outputs[i][j] - outputLayer[j]);
                deltaOutput[j] = errorOutput * dSigmoid(outputLayer[j]);
            }       
            // Compute error in the new hidden layer
            double deltaHidden3[numHiddenNodes3];
            for (int j = 0; j < numHiddenNodes3; j++) {
                double errorNewHidden = 0.0f;
                for (int k = 0; k < numOutputs; k++) {
                    errorNewHidden += deltaOutput[k] * outputWeights[j][k];
                }
                deltaHidden3[j] = errorNewHidden * dRelu(hiddenLayer3[j]);
            }
            // Compute error in the new hidden layer
            double deltaHidden2[numHiddenNodes2];
            for (int j = 0; j < numHiddenNodes2; j++) {
                double errorNewHidden = 0.0f;
                for (int k = 0; k < numHiddenNodes3; k++) {
                    errorNewHidden += deltaHidden3[k] * hiddenWeights3[j][k];
                }
                deltaHidden2[j] = errorNewHidden * dRelu(hiddenLayer2[j]);
            }
            // Compute change in hidden weights
            double deltaHidden[numHiddenNodes];
            for (int j=0; j<numHiddenNodes; j++) {
                double errorHidden = 0.0f;
                for(int k=0; k<numHiddenNodes2; k++) {
                    errorHidden += deltaHidden2[k] * hiddenWeights2[j][k];
                }
                deltaHidden[j] = errorHidden * dRelu(hiddenLayer[j]);
            }
            // Apply change in output weights
            for (int j=0; j<numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j] * lr;
                for (int k=0; k<numHiddenNodes3; k++) {
                    outputWeights[k][j] += hiddenLayer3[k] * deltaOutput[j] * lr;
                }
                
            }
            // Apply change in new hidden weights
            for (int j = 0; j < numHiddenNodes3; j++) {
                hiddenLayer3Bias[j] += deltaHidden3[j] * lr;
                for (int k = 0; k < numHiddenNodes2; k++) {
                    hiddenWeights3[k][j] += hiddenLayer2[k] * deltaHidden3[j] * lr;
                }
            }

            // Apply change in new hidden weights
            for (int j = 0; j < numHiddenNodes2; j++) {
                hiddenLayer2Bias[j] += deltaHidden2[j] * lr;
                for (int k = 0; k < numHiddenNodes; k++) {
                    hiddenWeights2[k][j] += hiddenLayer[k] * deltaHidden2[j] * lr;
                }
            }
            
            // Apply change in hidden weights
            for (int j=0; j<numHiddenNodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j] * lr;
                for(int k=0; k<numInputs; k++) {
                    hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * lr;
                }
            }

            // Loss calculation
            total_loss += cross_entropy_loss(training_outputs[i], outputLayer, numOutputs);

            // Storing predictions
            for (int j = 0; j < numOutputs; j++) {
                predictions[i][j] = outputLayer[j];
            }

        }

        // Calculate average loss and accuracy for the epoch
        double average_loss = total_loss / num_images;
        double accuracy = calculate_accuracy(training_outputs, predictions, num_images, numOutputs);

        // Print metrics
        printf("Epoch %d: Loss = %f, Accuracy = %f\n", epochs, average_loss, accuracy);
    }

    return 0;

}