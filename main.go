package main

import (
	"bufio"
	"encoding/csv"
	"io"
	"log"
	"os"
	"strconv"

	"github.com/fr3fou/gone/gone"
)

func main() {
	g := gone.New(
		0.01,
		gone.MGBD(20),
		gone.Layer{
			Nodes: 784,
		},
		gone.Layer{
			Nodes:     16,
			Activator: gone.ReLU(),
		},
		gone.Layer{
			Nodes:     16,
			Activator: gone.ReLU(),
		},
		gone.Layer{
			Nodes:     10,
			Activator: gone.Sigmoid(), // TODO: Softmax
		},
	)

	data := gone.DataSet{}

	csvFile, err := os.Open("train.csv")
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

	g.Train(data, 10)
}
