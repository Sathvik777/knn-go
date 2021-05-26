package main

import (
	"math"
	"sort"
)

type TrainData struct {
	Label    string
	Features []float64
}

type FeatureDistance struct {
	key      string
	distance float64
}

type ByDistance []FeatureDistance

func (a ByDistance) Len() int           { return len(a) }
func (a ByDistance) Less(i, j int) bool { return a[i].distance < a[j].distance }
func (a ByDistance) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

type KNNRegression struct {
	trainData []TrainData
}

func (knn *KNNRegression) nearestNeighbors(predictFeatures []float64) []FeatureDistance {
	distances := make([]FeatureDistance, len(knn.trainData))
	
	for i, row := range knn.trainData {
		sumOfSquared := 0.0
		for j, pf := range predictFeatures {
			point1 := row.Features[j]
			point2 := pf
			// If you want sound fancy you can say we are using Euclidean distance
			sumOfSquared += math.Pow(point1-point2, 2)
		}
		distances[i].key = row.Label
		distances[i].distance = math.Sqrt(sumOfSquared)
	}
	
	return distances
}

// Predict:
// feature of the current entity
// k number of neighbours you want to find
// returns k sized list of labels close.

func (knn *KNNRegression) Predict(features []float64, k int) []FeatureDistance {
	distances := knn.nearestNeighbors(features)
	sort.Sort(ByDistance(distances))
	return distances[:k]
}

func NewKNNRegression(trainData []TrainData) *KNNRegression {
	return &KNNRegression{
		trainData: trainData,
	}
	
}
