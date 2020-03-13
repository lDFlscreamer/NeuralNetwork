/*
 * Copyright (c)  3.2020
 * This file (NeuralNetwork) is part of NeuralNetwork.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Screamer  <999screamer999@gmail.com>
 */

package NeuralNetwork;

import NeuralNetwork.config.NetworkConfig;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

public class NeuralNetwork implements Serializable {
	private final Layer lastLayer;
	private Layer firstLayer;

	public NeuralNetwork(NetworkConfig config) {
		List<Integer> neuronsOnLayer = new ArrayList<>(config.getNeuronsOnLayer());
		neuronsOnLayer.add(config.getLastLayerNeuronsAmount());
		Layer current = null;
		for (Integer amount :
				neuronsOnLayer) {
			current = current == null ?
					new Layer(config.getInputArraySize(), amount, config.getFunc())
					: new Layer(current, amount, config.getFunc());
			if (firstLayer == null) {
				firstLayer = current;
			}
			if (current.getPreviousLayer() != null) {
				current.getPreviousLayer().setNextLayer(current);
			}
		}
		lastLayer = current;
	}

	public Layer getFirstLayer() {
		return firstLayer;
	}

	public void setFirstLayer(Layer firstLayer) {
		this.firstLayer = firstLayer;
	}

	public Layer getLastLayer() {
		return lastLayer;
	}

	public Map<Integer, Double> calculate(double[] input) throws ExecutionException, InterruptedException {
		Layer layer = firstLayer;

		Map<Integer, Double> layerResult = new HashMap<>();
		for (int i = 0; i < input.length; i++) {
			layerResult.put(i, input[i]);
		}

		while (layer != null) {
			layer.setPreviousLayerResults(layerResult);
			layerResult = layer.calculate();
			layer = layer.getNextLayer();
		}
		return layerResult;
	}

}
