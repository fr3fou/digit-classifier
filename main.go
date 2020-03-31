package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"

	"github.com/fr3fou/gone/gone"
)

func main() {
	g := gone.New(
		0.01,
		gone.Layer{
			Nodes: 784,
		},
		gone.Layer{
			Nodes:     20,
			Activator: gone.ReLU(),
		},
		gone.Layer{
			Nodes:     10,
			Activator: gone.Sigmoid(), // TODO: Softmax
		},
	)

	log.Println("Parsing  csv...")
	data := parse("test.csv")
	log.Println("Finished parsing csv...")

	g.Train(
		gone.MGBD(20),
		data,
		1,
	)
}

func test(testName string, outputName string, n *gone.NeuralNetwork) {
	testFile, err := os.Open(testName)
	if err != nil {
		panic(err)
	}

	outputFile, err := os.Create(outputName)
	if err != nil {
		panic(err)
	}

	isHeader := true
	reader := csv.NewReader(bufio.NewReader(testFile))

	id := 1

	outputFile.WriteString("ImageId,Label")
	for {
		line, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}

		// Skip the header
		if isHeader {
			isHeader = false
			continue
		}

		pixels := []float64{}
		for i := 1; i < len(line); i++ {
			pixel, err := strconv.ParseFloat(line[i], 64)
			if err != nil {
				log.Println(err)
				continue
			}
			pixels = append(pixels, pixel)
		}

		labels := n.Predict(pixels)
		bestScore := -1.0
		bestDigit := 0.0
		for digit, score := range labels {
			if score > bestScore {
				bestScore = score
				bestDigit = float64(digit)
			}
		}

		outputFile.WriteString(fmt.Sprintf("%d,%d\n", int(id), int(bestDigit)))
		id++
	}
}

func parse(file string) gone.DataSet {
	data := gone.DataSet{}

	csvFile, err := os.Open(file)
	if err != nil {
		panic(err)
	}

	isHeader := true
	reader := csv.NewReader(bufio.NewReader(csvFile))
	for {
		line, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}

		// Skip the header
		if isHeader {
			isHeader = false
			continue
		}

		label, err := strconv.ParseFloat(line[0], 64)
		if err != nil {
			log.Println(err)
			continue
		}

		labels := [10]float64{}
		labels[int(label)] = 1.0 // set only the correct label to 1.0 and keep the rest at 0.0

		pixels := []float64{}
		for i := 1; i < len(line); i++ {
			pixel, err := strconv.ParseFloat(line[i], 64)
			if err != nil {
				log.Println(err)
				continue
			}
			pixels = append(pixels, pixel)
		}

		data = append(data, gone.DataSample{
			Inputs:  pixels,
			Targets: labels[:],
		})
	}

	if err := csvFile.Close(); err != nil {
		panic(err)
	}

	return data
}
