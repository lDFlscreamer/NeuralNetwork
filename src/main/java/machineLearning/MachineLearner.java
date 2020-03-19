/*
 * Copyright (c)  3.2020
 * This file (BackPropagation) is part of NeuralNetwork.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Screamer  <999screamer999@gmail.com>
 */

package machineLearning;

import neuralNetwork.NeuralNetwork;

public class MachineLearner {
	protected final NeuralNetwork neuralNetwork;
	protected final Cost costFunction;
	protected final double learningRate;

	public MachineLearner(NeuralNetwork neuralNetwork, Cost costFunction, double learningRate) {
		this.neuralNetwork = neuralNetwork;
		this.costFunction = costFunction;
		this.learningRate = learningRate;
	}

}
