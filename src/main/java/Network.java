/*
 * Copyright (c)  3.2020
 * This file (Network) is part of NeuralNetwork.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Screamer  <999screamer999@gmail.com>
 */

import config.NetworkConfig;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;

public class Network implements Serializable {
	private Layer firstLayer;
	private Layer lastLayer;

	public Network(NetworkConfig config) {
		List<Integer> neuronsOnLayer = config.getNeuronsOnLayer();
		Layer current = null;
		for (Integer amount :
				neuronsOnLayer) {
			current = current==null?
					new Layer(config.getInputArraySize(),amount,config.getFunc())
					:new Layer(current, amount, config.getFunc());
			if (firstLayer == null) {
				firstLayer = current;
			}
			if (current.getPreviousLayer() != null) {
				current.getPreviousLayer().setNextLayer(current);
			}
		}
		lastLayer = current;
	}


	public Map<Integer, Double> calculate(double[] input) throws ExecutionException, InterruptedException {
		Layer layer = firstLayer;

		Map<Integer, Double> layerResult = new HashMap<>();
		for (int i = 0; i < input.length; i++) {
			layerResult.put(i, input[i]);
		}

		while (layer!=null){
			layer.setPreviousLayerResults(layerResult);
			layerResult = layer.calculate();
			layer=layer.getNextLayer();
		}
		return layerResult;
	}

}
