/*
 * Copyright (c)  3.2020
 * This file (Layer) is part of NeuralNetwork.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Screamer  <999screamer999@gmail.com>
 */

package neuralNetwork;

import neuralNetwork.config.Function;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;

public class Layer implements Serializable {
	private final Function function;
	private final Layer previousLayer;
	private final List<Neuron> neurons;
	private final Map<Integer, Double> previousLayerResults;
	private final Map<Integer, Double> layerResults;
	private Layer nextLayer;

	public Layer(Layer previousLayer, int initialNeuronAmount, Function function) {
		this.previousLayer = previousLayer;
		this.previousLayerResults = new HashMap<>();
		this.layerResults = new HashMap<>();

		this.function = function;
		this.neurons = new ArrayList<>();
		for (int i = 0; i < initialNeuronAmount; i++) {
			neurons.add(new Neuron(previousLayer.getNeuronsAmount(), function));
		}
	}

	public Layer(int previousLayerNeuronAmount, int initialNeuronAmount, Function function) {
		this.previousLayer = null;
		this.previousLayerResults = new HashMap<>();
		this.layerResults = new HashMap<>();

		this.function = function;
		this.neurons = new ArrayList<>();
		for (int i = 0; i < initialNeuronAmount; i++) {
			neurons.add(new Neuron(previousLayerNeuronAmount, function));
		}
	}

	public List<Neuron> getNeurons() {
		return neurons;
	}

	public Function getFunction() {
		return function;
	}

	public Map<Integer, Double> getPreviousLayerResults() {
		return previousLayerResults;
	}

	public void setPreviousLayerResults(Map<Integer, Double> previousLayerResults) {
		this.previousLayerResults.clear();
		this.previousLayerResults.putAll(previousLayerResults);
	}

	public void setPreviousLayerResults(final double[] previousLayerResults) {
		this.previousLayerResults.clear();
		for (int i = 0; i < previousLayerResults.length; i++) {
			this.previousLayerResults.put(i, previousLayerResults[i]);
		}
	}

	public Layer getPreviousLayer() {
		return previousLayer;
	}

	public Layer getNextLayer() {
		return nextLayer;
	}

	public void setNextLayer(Layer nextLayer) {
		this.nextLayer = nextLayer;
	}

	public Map<Integer, Double> getLayerResults() {
		return layerResults;
	}

	public int getNeuronsAmount() {
		return neurons.size();
	}

	/**
	 * Calculate neuron signal on this layer
	 * Need to previousLayerResults have been filled
	 *
	 * @return resulted tensor
	 * @throws ExecutionException   may cause when neuron.get
	 * @throws InterruptedException may cause when neuron.get
	 */
	public Map<Integer, Double> calculate() throws ExecutionException, InterruptedException {
		ForkJoinPool forkJoinPool = new ForkJoinPool();
		List<ForkJoinTask<Double>> tasks = new ArrayList<>();
		for (Neuron n :
				neurons) {
			n.setPreviousLayerResults(this.previousLayerResults);
			Callable<Double> calculateActivationFunction = n::calculateActivationFunction;
			ForkJoinTask<Double> task = forkJoinPool.submit(calculateActivationFunction);
			tasks.add(task);
		}

		this.layerResults.clear();
		boolean allDone;
		do {
			allDone = true;
			for (int i = 0; i < tasks.size(); i++) {
				ForkJoinTask<Double> task = tasks.get(i);
				allDone &= task.isDone();
				if (task.isDone()) {
					layerResults.put(i, task.get());
				}
			}

		} while (!allDone);

		return this.layerResults;
	}

}
