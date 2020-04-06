package main

import (
	"fmt"
	"net/http"

	"github.com/fr3fou/gone/gone"
	"github.com/go-chi/chi"
)

// Web is a web struct
type Web struct {
	G *gone.NeuralNetwork
	R *chi.Mux
}

// NewWeb is a constructor for Web
func NewWeb(g *gone.NeuralNetwork) *Web {
	r := chi.NewRouter()
	w := &Web{g, r}
	r.Post("/guess", w.guess)
	return w
}

func (web *Web) guess(w http.ResponseWriter, r *http.Request) {
	r.ParseMultipartForm(1 << 20) // 1 MiB
	f, header, err := r.FormFile("file")
	if err != nil || header.Size > 1<<20 {
		fmt.Fprintf(w, "Max file size is 1 MiB")
		return
	}

	image, err := imageToBytes(f)
	if err != nil {
		fmt.Fprintf(w, err.Error())
	}

	guess := squash(web.G.Predict(image))
	fmt.Fprintf(w, "%d\n", guess)
}

func (web Web) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	web.R.ServeHTTP(w, r)
}
