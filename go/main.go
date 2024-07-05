package main

import (
    "encoding/json"
    "fmt"
    "math/rand"
    "time"
)

func main() {
    // Set hyperparameters
    layerSizes := []int{784, 128, 64, 10}
    learningRate := 0.1
    epochs := 30
    miniBatchSize := 32

    // Set fixed seed for reproducibility
    rand.Seed(42)

    fmt.Println("Loading MNIST data...")
    startTime := time.Now()

    // Load and preprocess data
    trainImages, err := loadMNISTImages("data/train-images-idx3-ubyte")
    if err != nil {
        fmt.Println("Error loading training images:", err)
        return
    }
    trainLabels, err := loadMNISTLabels("data/train-labels-idx1-ubyte")
    if err != nil {
        fmt.Println("Error loading training labels:", err)
        return
    }
    testImages, err := loadMNISTImages("data/t10k-images-idx3-ubyte")
    if err != nil {
        fmt.Println("Error loading test images:", err)
        return
    }
    testLabels, err := loadMNISTLabels("data/t10k-labels-idx1-ubyte")
    if err != nil {
        fmt.Println("Error loading test labels:", err)
        return
    }

    trainLabelsEncoded := oneHotEncode(trainLabels, 10)
    testLabelsEncoded := oneHotEncode(testLabels, 10)

    // Shuffle training data
    trainingData := make([]struct{X, Y []float64}, len(trainImages))
    for i := range trainImages {
        trainingData[i] = struct{X, Y []float64}{trainImages[i], trainLabelsEncoded[i]}
    }
    rand.Shuffle(len(trainingData), func(i, j int) { trainingData[i], trainingData[j] = trainingData[j], trainingData[i] })