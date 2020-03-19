/*
 * Copyright (c)  3.2020
 * This file (BackPropagation) is part of NeuralNetwork.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Screamer  <999screamer999@gmail.com>
 */

package machineLearning.parallel.backPropagation;

import machineLearning.Cost;
import machineLearning.MachineLearner;
import machineLearning.learningData.LearningSample;
import machineLearning.parallel.MachineLearnerParallelInterface;
import neuralNetwork.Layer;
import neuralNetwork.NeuralNetwork;
import neuralNetwork.Neuron;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;

public class BackPropagationParallel extends MachineLearner implements MachineLearnerParallelInterface {


	public BackPropagationParallel(NeuralNetwork neuralNetwork, Cost costFunction, double learningRate) {
		super(neuralNetwork, costFunction, learningRate);
	}

	public List<ForkJoinTask<Void>> computeAdjust(double[] input, double[] ideal, ConcurrentMap<Integer, Double> biasAdjust, ConcurrentMap<Integer, ConcurrentMap<Integer, Double>> weightAdjust) throws ExecutionException, InterruptedException {
		neuralNetwork.calculate(input);
		ForkJoinPool forkJoinPool = new ForkJoinPool();
		Layer layer = neuralNetwork.getLastLayer();
		weightAdjust = weightAdjust == null ? new ConcurrentHashMap<>() : weightAdjust;
		biasAdjust = biasAdjust == null ? new ConcurrentHashMap<>() : biasAdjust;
		List<ForkJoinTask<Void>>tasks=new ArrayList<>();
		while (layer != null) {

			for (int i = 0; i < layer.getLayerResults().size(); i++) {
				tasks.add(forkJoinPool.submit(new BackPropagationAdjustFinderTask(biasAdjust, weightAdjust, ideal, layer, i)));
			}
			layer = layer.getPreviousLayer();
		}
		return  tasks;
	}

	public void learnNetwork(List<LearningSample> data){
		ConcurrentMap<Integer, Double> biasAdjust = new ConcurrentHashMap<>();
		ConcurrentMap<Integer, ConcurrentMap<Integer,Double>> wightAdjust = new ConcurrentHashMap<>();
		List<ForkJoinTask<Void>> tasks=new ArrayList<>();
		for (LearningSample sample :
				data) {
			try {
				tasks.addAll(computeAdjust(sample.getInput(), sample.getIdeal(), biasAdjust, wightAdjust));
			} catch (ExecutionException | InterruptedException e) {
				e.printStackTrace();
			}
		}
		boolean done =false;
		while (!done){
			done=tasks.stream().map(ForkJoinTask::isDone).reduce((aBoolean, aBoolean2) -> aBoolean&aBoolean2).orElse(true);
		}
		Layer layer=neuralNetwork.getFirstLayer();
		while (layer!=null){
			for (Neuron n :
					layer.getNeurons()) {
				Map<Integer, Double> currentWeightAdjust = wightAdjust.get(n.getId());
				List<Double> weights = n.getWeights();
				for (Integer key :
						currentWeightAdjust.keySet()) {
					Double currentWeight = weights.get(key);
					weights.set(key,currentWeight-(learningRate*(currentWeightAdjust.get(key)/data.size())));
				}
				n.setBias(n.getBias()-(learningRate*(biasAdjust.get(n.getId())/data.size())));
			}
			layer=layer.getNextLayer();
		}
	}


}
