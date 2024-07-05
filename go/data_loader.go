package main

import (
    "encoding/binary"
    "os"
)

func loadMNISTImages(filename string) ([][]float64, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    var magic, numImages, numRows, numCols int32
    binary.Read(file, binary.BigEndian, &magic)
    binary.Read(file, binary.BigEndian, &numImages)
    binary.Read(file, binary.BigEndian, &numRows)
    binary.Read(file, binary.BigEndian, &numCols)

    images := make([][]float64, numImages)
    for i := range images {
        images[i] = make([]float64, numRows*numCols)
        for j := range images[i] {
            var pixel byte
            binary.Read(file, binary.BigEndian, &pixel)
            images[i][j] = float64(pixel) / 255.0
        }
    }

    return images, nil
}

func loadMNISTLabels(filename string) ([]int, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    var magic, numLabels int32
    binary.Read(file, binary.BigEndian, &magic)
    binary.Read(file, binary.BigEndian, &numLabels)

    labels := make([]int, numLabels)
    for i := range labels {
        var label byte
        binary.Read(file, binary.BigEndian, &label)
        labels[i] = int(label)
    }

    return labels, nil
}

func oneHotEncode(labels []int, numClasses int) [][]float64 {
    encoded := make([][]float64, len(labels))
    for i, label := range labels {
        encoded[i] = make([]float64, numClasses)
        encoded[i][label] = 1.0
    }
    return encoded
}