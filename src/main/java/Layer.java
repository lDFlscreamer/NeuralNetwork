/*
 * Copyright (c)  3.2020
 * This file (Layer) is part of NeuralNetwork.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Screamer  <999screamer999@gmail.com>
 */

import config.Function;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

public class Layer extends RecursiveTask<Map<Integer, Double>>implements Serializable {
	private final Function function;
	private final Layer previousLayer;
	private final List<Neuron> neurons;
	private final Map<Integer, Double> previousLayerResults;
	private Layer nextLayer;
	private final Map<Integer, Double> layerResults;

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
		this.previousLayer=null;
		this.previousLayerResults = new HashMap<>();
		this.layerResults = new HashMap<>();

		this.function = function;
		this.neurons = new ArrayList<>();
		for (int i = 0; i < initialNeuronAmount; i++) {
			neurons.add(new Neuron(previousLayerNeuronAmount, function));
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

	/**
	 * Calculate neuron signal on this layer
	 * Need to previousLayerResults have been filled
	 *
	 * @return resulted tensor
	 * @throws ExecutionException may cause when neuron.get
	 * @throws InterruptedException may cause when neuron.get
	 */
	public Map<Integer, Double> calculate() throws ExecutionException, InterruptedException {
		ForkJoinPool forkJoinPool=new ForkJoinPool();
		for (Neuron n :
				neurons) {
			n.setPreviousLayerResults(this.previousLayerResults);
			forkJoinPool.submit(n);
		}

		this.layerResults.clear();
		boolean allDone;
		do {
			allDone=true;
			for (int i = 0; i < neurons.size(); i++) {
				Neuron neuron = neurons.get(i);
				allDone &= neuron.isDone();
				if (neuron.isDone()) {
					layerResults.put(i, neuron.get());
				}
			}

		} while (!allDone);

		return this.layerResults;
	}

	@Override
	protected Map<Integer, Double> compute() {
		try {
			return calculate();
		} catch (ExecutionException | InterruptedException e) {
			e.printStackTrace();
		}
		return null;
	}
}