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
		0.1,
		gone.Layer{
			Nodes: 784,
		},
		gone.Layer{
			Nodes:     20,
			Activator: gone.Sigmoid(),
		},
		gone.Layer{
			Nodes: 10,
			// Activator: gone.Sigmoid(), // TODO: Softmax
		},
	)
	g.ToggleDebug(true)

	log.Println("Parsing csv...")
	data := parse("train.csv")
	log.Println("Finished parsing csv...")

	log.Println("Beginning training ...")
	g.Train(
		gone.SGD(),
		data,
		10,
	)
	log.Println("Finished training ...")

	log.Println("Writing out.csv...")
	test("test.csv", "out.csv", g)
}

func test(testName string, outputName string, n *gone.NeuralNetwork) {
	testFile, err := os.Open(testName)
	if err != nil {
		panic(err)
	}
	defer testFile.Close()

	outputFile, err := os.Create(outputName)
	if err != nil {
		panic(err)
	}
	defer outputFile.Close()

	isHeader := true
	reader := csv.NewReader(bufio.NewReader(testFile))

	id := 1

	outputFile.WriteString("ImageId,Label\n")
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
		for i := 0; i < len(line); i++ {
			pixel, err := strconv.ParseFloat(line[i], 64)
			if err != nil {
				log.Println(err)
				continue
			}
			pixels = append(pixels, pixel/255)
		}

		labels := n.Predict(pixels)
		bestScore := 0.0
		bestDigit := 0.0
		for digit, score := range labels {
			if score > bestScore {
				bestScore = score
				bestDigit = float64(digit)
			}
		}

		outputFile.WriteString(fmt.Sprintf("%d,%d\n", id, int(bestDigit)))
		id++
	}
}

func parse(file string) gone.DataSet {
	data := gone.DataSet{}

	csvFile, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer csvFile.Close()

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
			pixels = append(pixels, pixel/255)
		}

		data = append(data, gone.DataSample{
			Inputs:  pixels,
			Targets: labels[:],
		})
	}

	return data
}
