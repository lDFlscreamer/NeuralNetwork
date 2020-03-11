/*
 * Copyright (c)  3.2020
 * This file (NetworkMain) is part of NeuralNetwork.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Screamer  <999screamer999@gmail.com>
 */


import config.NetworkConfig;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

public class NetworkMain {

	public static void main(String[] args) throws ExecutionException, InterruptedException {
		NetworkConfig config=new NetworkConfig();
		config.setFunc(NetworkConfig.SIGMOID);
		config.setInputArraySize(2);
		List<Integer> integers = new ArrayList<>();
		integers.add(3);
		integers.add(6);
		integers.add(8);
		config.setNeuronsOnLayer(integers);

		Network network=new Network(config);
		double[] doubles = {0.1, 0.5};
		Map<Integer, Double> calculate = network.calculate(doubles);
		System.out.println("calculate = " + calculate);
	}
}
