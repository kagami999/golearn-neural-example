package main

import (
	"fmt"
	"log"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/neural"
)

func main() {
	layers := []int{4, 16, 3}
	network := neural.NewMultiLayerNet(layers)
	inst, e := base.ParseCSVToInstances("iris_headers.csv",
		true)
	if e != nil {
		log.Fatal(e)
	}
	c_attribute := base.GetAttributeByName(inst,
		"Species")
	e = inst.AddClassAttribute(c_attribute)
	if e != nil {
		log.Fatal(e)
	}
	train, test := base.InstancesTrainTestSplit(inst,
		0.80)
	network.MaxIterations = 10000
	network.Convergence = 0.001
	network.LearningRate = 0.001
	network.Fit(train)
	pred := network.Predict(test)
	cm, e := evaluation.GetConfusionMatrix(test,
		pred)
	if e != nil {
		log.Fatal(e)
	}
	acc := evaluation.GetAccuracy(cm)
	fmt.Println(acc)
}
