package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"image"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	_ "image/png"

	"github.com/fr3fou/gone/gone"
)

func main() {
	g, err := gone.Load("95%.gone")
	if err != nil {
		panic(err)
	}

	w := NewWeb(g)
	log.Println("listening on port :8080")
	if err := http.ListenAndServe(":8080", w); err != nil {
		panic(err)
	}
}

func _main() {
	g := gone.New(
		0.1,
		gone.MSE(),
		gone.Layer{
			Nodes: 784,
		},
		gone.Layer{
			Nodes:     20,
			Activator: gone.Sigmoid(),
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
	g.SetDebugMode(true)

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

	t := strconv.FormatInt(time.Now().Unix(), 10)
	g.Save("digit-" + t + ".gone")

	log.Println("Writing out.csv...")
	test("test.csv", "out-"+t+".csv", g)

}

func loadImage(imageName string) ([]float64, error) {
	file, err := os.Open(imageName)
	if err != nil {
		return nil, err
	}

	return imageToBytes(file)

}

func imageToBytes(r io.Reader) ([]float64, error) {
	img, _, err := image.Decode(r)
	if err != nil {
		return nil, err
	}

	// Converting image to grayscale
	grayImg := image.NewGray(img.Bounds())
	for y := img.Bounds().Min.Y; y < img.Bounds().Max.Y; y++ {
		for x := img.Bounds().Min.X; x < img.Bounds().Max.X; x++ {
			grayImg.Set(x, y, img.At(x, y))
		}
	}

	v := []float64{}
	for _, pixel := range grayImg.Pix {
		v = append(v, float64(pixel))
	}
	return v, nil
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
		bestDigit := squash(labels)

		outputFile.WriteString(fmt.Sprintf("%d,%d\n", id, bestDigit))
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

func squash(predictions []float64) int {
	bestScore := 0.0
	bestDigit := 0
	for digit, score := range predictions {
		if score > bestScore {
			bestScore = score
			bestDigit = digit
		}
	}

	return bestDigit
}
