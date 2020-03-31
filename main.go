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

type Digit struct {
	Pixels []float64
	Label  float64
}

func main() {
	gone.New(
		0.01,
		gone.SGD(),
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

	digits := []Digit{}

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

		pixels := []float64{}
		for i := 1; i < len(line); i++ {
			pixel, err := strconv.ParseFloat(line[i], 64)
			if err != nil {
				log.Println(err)
				continue
			}
			pixels = append(pixels, pixel)
		}

		digits = append(digits, Digit{
			Label:  label,
			Pixels: pixels,
		})
	}
}
