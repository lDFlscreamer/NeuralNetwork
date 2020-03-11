/*
 * Copyright (c)  3.2020
 * This file (Neuron) is part of NeuralNetwork.
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
import java.util.concurrent.RecursiveTask;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;

public class Neuron extends RecursiveTask<Double> implements Serializable {
	private static final AtomicInteger amount = new AtomicInteger(0);
	private final List<Double> weights;
	private final int id;
	private final Function function;
	private final Map<Integer, Double> previousLayerResults;
	private double bias;


	public Neuron(Function function) {
		this(1, function);
		this.changeWeights(0, 1);
		this.setBias(0);
	}

	public Neuron(int initialWeightAmount, Function function) {
		this.id = amount.getAndIncrement();
		this.previousLayerResults = new HashMap<>();
		this.function = function;
		this.bias = ThreadLocalRandom.current().nextDouble();

		this.weights = new ArrayList<>();
		for (int i = 0; i < initialWeightAmount; i++) {
			weights.add(ThreadLocalRandom.current().nextDouble() + 0.1);
		}
	}

	public List<Double> getWeights() {
		return weights;
	}

	public void changeWeights(int index, double value) throws IndexOutOfBoundsException {
		if (index > weights.size()) {
			throw new IndexOutOfBoundsException();
		}
		weights.set(index, value);
	}

	public double getBias() {
		return bias;
	}

	public void setBias(double bias) {
		this.bias = bias;
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

	@Override
	protected Double compute() {
		System.out.println("Neuron.compute");
		double sum = 0;
		for (int i = 0; i < previousLayerResults.size(); i++) {
			sum += this.previousLayerResults.get(i) * this.weights.get(i);
		}
		sum += bias;
		return this.function.calculate(sum);
	}
}
