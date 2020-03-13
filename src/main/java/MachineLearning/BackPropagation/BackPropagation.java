/*
 * Copyright (c)  3.2020
 * This file (BackPropagation) is part of NeuralNetwork.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Screamer  <999screamer999@gmail.com>
 */

package MachineLearning.BackPropagation;

import MachineLearning.Cost;
import MachineLearning.LearningData.LearningSample;
import MachineLearning.MachineLearner;
import NeuralNetwork.Layer;
import NeuralNetwork.NeuralNetwork;
import NeuralNetwork.Neuron;
import NeuralNetwork.config.Function;
import NeuralNetwork.config.Functions;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

public class BackPropagation extends MachineLearner implements BackPropagationInterface {


	public BackPropagation(NeuralNetwork neuralNetwork, Cost costFunction, double learningRate) {
		super(neuralNetwork, costFunction, learningRate);
	}

	public void computeAdjust(double[] input, double[] ideal, Map<Integer, Double> biasAdjust, Map<Integer, Map<Integer, Double>> weightAdjust) throws ExecutionException, InterruptedException {
		neuralNetwork.calculate(input);
		Layer layer = neuralNetwork.getLastLayer();
		Map<Integer, Double> lastLayerResult = layer.getLayerResults();
		weightAdjust = weightAdjust == null ? new HashMap<>() : weightAdjust;
		biasAdjust = biasAdjust == null ? new HashMap<>() : biasAdjust;
		double biasAdjustValue,
				dCost_dSignal,
				functionDerivative;
		while (layer != null) {
			for (int i = 0; i < layer.getLayerResults().size(); i++) {
				Neuron n = layer.getNeurons().get(i);

				dCost_dSignal = dCost_dSignal(layer, i, ideal);

				functionDerivative = functionDerivative(layer.getFunction(), layer.getLayerResults().get(i));
				biasAdjustValue = dCost_dSignal * functionDerivative;

				double biasValue = biasAdjust.getOrDefault(n.getId(), 0.0) + biasAdjustValue;
				biasAdjust.put(n.getId(), biasValue);
				for (int j = 0; j < n.getWeights().size(); j++) {
					double dSignal_dweight = layer.getPreviousLayerResults().get(j);

					Map<Integer, Double> current = weightAdjust.getOrDefault(n.getId(), new HashMap<>());
					double weightValue = current.getOrDefault(j, 0.0) + (biasAdjustValue * dSignal_dweight);
					current.put(j, weightValue);
					weightAdjust.put(n.getId(), current);
				}

			}

			layer = layer.getPreviousLayer();
		}

	}

	public void learnNetwork(List<LearningSample> data){
		HashMap<Integer, Double> biasAdjust = new HashMap<>();
		HashMap<Integer, Map<Integer,Double>> wightAdjust = new HashMap<>();
		for (LearningSample sample :
				data) {
			try {
				computeAdjust(sample.getInput(), sample.getIdeal(), biasAdjust, wightAdjust);
			} catch (ExecutionException | InterruptedException e) {
				e.printStackTrace();
			}
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


	public double functionDerivative(Function func, double sum) {
		Function derivative = Functions.DERIVATIVES.getOrDefault(func, s -> s);
		return derivative.calculate(sum);
	}

	public double dCost_dSignal(Layer current, Integer indexOfNeuron, double[] ideal) {
		if (current.getNextLayer() == null) {

			return (2 * (current.getLayerResults().get(indexOfNeuron) - ideal[indexOfNeuron]));
		}
		Layer nextLayer = current.getNextLayer();
		double derivative = 0;
		double v;
		for (int i = 0; i < nextLayer.getNeuronsAmount(); i++) {
			v = nextLayer.getNeurons().get(i).getWeights().get(indexOfNeuron);
			v *= dCost_dSignal(nextLayer, i, ideal) * functionDerivative(nextLayer.getFunction(), nextLayer.getLayerResults().get(i));
			derivative += v;
		}
		return derivative;
	}
}
