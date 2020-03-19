/*
 * Copyright (c)  3.2020
 * This file (NetworkConfig) is part of NeuralNetwork.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Screamer  <999screamer999@gmail.com>
 */

package neuralNetwork.config;

import java.util.ArrayList;
import java.util.List;

public class NetworkConfig {
	private List<Integer> neuronsOnLayer;
	private Integer inputArraySize;
	private Integer lastLayerNeuronsAmount;
	private Function func;

	public NetworkConfig() {
		this(0, new ArrayList<>(), 0, Functions.SIGMOID);
	}
	public NetworkConfig(Integer inputArraySize, List<Integer> neuronsOnLayer, Integer lastLayerNeuronsAmount) {
		this(inputArraySize, neuronsOnLayer, lastLayerNeuronsAmount, Functions.SIGMOID);
	}

	public NetworkConfig(Integer inputArraySize, List<Integer> neuronsOnLayer, Integer lastLayerNeuronsAmount, Function func) {
		this.neuronsOnLayer = neuronsOnLayer;
		this.inputArraySize = inputArraySize;
		this.lastLayerNeuronsAmount = lastLayerNeuronsAmount;
		this.func = func;
	}

	public Integer getLastLayerNeuronsAmount() {
		return lastLayerNeuronsAmount;
	}

	public void setLastLayerNeuronsAmount(Integer lastLayerNeuronsAmount) {
		this.lastLayerNeuronsAmount = lastLayerNeuronsAmount;
	}

	public Integer getInputArraySize() {
		return inputArraySize;
	}

	public void setInputArraySize(Integer inputArraySize) {
		this.inputArraySize = inputArraySize;
	}

	public List<Integer> getNeuronsOnLayer() {
		return neuronsOnLayer;
	}

	public void setNeuronsOnLayer(List<Integer> neuronsOnLayer) {
		this.neuronsOnLayer = neuronsOnLayer;
	}

	public Function getFunc() {
		return func;
	}

	public void setFunc(Function func) {
		this.func = func;
	}

}
