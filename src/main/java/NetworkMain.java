/*
 * Copyright (c)  3.2020
 * This file (NetworkMain) is part of NeuralNetwork.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Screamer  <999screamer999@gmail.com>
 */


import MachineLearning.CostFunctions;
import MachineLearning.LearningData.LearningSample;
import MachineLearning.BackPropagation.BackPropagation;
import NeuralNetwork.NeuralNetwork;
import NeuralNetwork.config.Functions;
import NeuralNetwork.config.NetworkConfig;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

public class NetworkMain {

	public static void main(String[] args) throws ExecutionException, InterruptedException {
		NetworkConfig config = new NetworkConfig();
/*create NeuralNetwork.config*/
		config.setFunc(Functions.SIGMOID);
		config.setInputArraySize(2);
		List<Integer> integers = new ArrayList<>();
		integers.add(2);
//		integers.add(8);
		config.setNeuronsOnLayer(integers);
		config.setLastLayerNeuronsAmount(2);
/*create Network*/
		NeuralNetwork network = new NeuralNetwork(config);
/*
		network.getFirstLayer().getNeurons().get(0).setBias(0.35);
		network.getFirstLayer().getNeurons().get(1).setBias(0.35);
		network.getFirstLayer().getNextLayer().getNeurons().get(0).setBias(0.60);
		network.getFirstLayer().getNextLayer().getNeurons().get(1).setBias(0.60);
		network.getFirstLayer().getNeurons().get(0).getWeights().set(0,0.15);
		network.getFirstLayer().getNeurons().get(0).getWeights().set(1,0.2);
		network.getFirstLayer().getNeurons().get(1).getWeights().set(0,0.25);
		network.getFirstLayer().getNeurons().get(1).getWeights().set(1,0.30);
		network.getFirstLayer().getNextLayer().getNeurons().get(0).getWeights().set(0,0.40);
		network.getFirstLayer().getNextLayer().getNeurons().get(0).getWeights().set(1,0.45);
		network.getFirstLayer().getNextLayer().getNeurons().get(1).getWeights().set(0,0.5);
		network.getFirstLayer().getNextLayer().getNeurons().get(1).getWeights().set(1,0.55);
*/





		double[] doubles = {0.05 ,0.1 };
		Map<Integer, Double> calculate = network.calculate(doubles);
		System.out.println("calculate = " + calculate);

		/*create Learning sample*/
		List<LearningSample> learningSampleList=new ArrayList<>();
		learningSampleList.add(new LearningSample(new double[]{0.05 ,0.10},new double[]{0.01, 0.99}));

		/*create Learner*/
		BackPropagation backPropagation =new BackPropagation(network, CostFunctions.COST, 0.5);

		for (int i = 0; i < 10; i++) {
			backPropagation.learnNetwork(learningSampleList);
		}

		calculate = network.calculate(doubles);
		System.out.println("calculate = " + calculate);

	}
}
