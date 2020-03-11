/*
 * Copyright (c)  3.2020
 * This file (MachineLearning) is part of NeuralNetwork.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Screamer  <999screamer999@gmail.com>
 */

import config.Cost;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;

public class MachineLearning {

	private static final Cost COST = (s, s1) -> Math.pow(s - s1, 2);
	private NeuralNetwork neuralNetwork;

	public Map<Integer, Double> calculate(double[] input) throws ExecutionException, InterruptedException {
		Layer layer = this.neuralNetwork.getFirstLayer();
		// TODO: 11.03.2020 write change weight and bias


		Map<Integer, Double> layerResult = new HashMap<>();
		for (int i = 0; i < input.length; i++) {
			layerResult.put(i, input[i]);
		}

		while (layer != null) {
			layer.setPreviousLayerResults(layerResult);
			layerResult = layer.calculate();
			layer=layer.getNextLayer();
		}
		return layerResult;
	}

}
