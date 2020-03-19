/*
 * Copyright (c)  3.2020
 * This file (BackPropagationWeightAdjustFinder) is part of NeuralNetwork.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Screamer  <999screamer999@gmail.com>
 */

package machineLearning.parallel.backPropagation;

import neuralNetwork.Layer;
import neuralNetwork.Neuron;
import neuralNetwork.config.Function;
import neuralNetwork.config.Functions;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class BackPropagationAdjustFinderTask extends RecursiveAction implements FunctionDerivative  {
	private final double[] ideal;
	private final Layer layer;
	private final Integer indexOfNeuron;
	private final ConcurrentMap<Integer, Double> biasAdjust;
	private final ConcurrentMap<Integer, ConcurrentMap<Integer, Double>> weightAdjust;

	public BackPropagationAdjustFinderTask(ConcurrentMap<Integer, Double> biasAdjust,
										   ConcurrentMap<Integer, ConcurrentMap<Integer, Double>> weightAdjust,
										   double[] ideal, Layer layer, Integer indexOfNeuron) {
		this.biasAdjust = biasAdjust;
		this.weightAdjust = weightAdjust;
		this.ideal = ideal;
		this.layer = layer;
		this.indexOfNeuron = indexOfNeuron;
	}

	@Override
	protected void compute() {
		Neuron n = layer.getNeurons().get(indexOfNeuron);

		double dCost_dSignal = dCost_dSignal(layer,indexOfNeuron,ideal);

		double functionDerivative = FunctionDerivative.functionDerivative(layer.getFunction(), layer.getLayerResults().get(indexOfNeuron));
		double biasAdjustValue = dCost_dSignal * functionDerivative;

		double biasValue = biasAdjust.getOrDefault(n.getId(), 0.0) + biasAdjustValue;
		biasAdjust.put(n.getId(), biasValue);
		for (int j = 0; j < n.getWeights().size(); j++) {
			double dSignal_dweight = layer.getPreviousLayerResults().get(j);

			ConcurrentMap<Integer, Double> current = weightAdjust.getOrDefault(n.getId(), new ConcurrentHashMap<>());
			double weightValue = current.getOrDefault(j, 0.0) + (biasAdjustValue * dSignal_dweight);
			current.put(j, weightValue);
			weightAdjust.put(n.getId(), current);
		}
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
			v *= dCost_dSignal(nextLayer, i, ideal) * FunctionDerivative.functionDerivative(nextLayer.getFunction(), nextLayer.getLayerResults().get(i));
			derivative += v;
		}
		return derivative;
	}


}
