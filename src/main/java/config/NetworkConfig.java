/*
 * Copyright (c)  3.2020
 * This file (NetworkConfig) is part of NeuralNetwork.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Screamer  <999screamer999@gmail.com>
 */

package config;/*
 * Copyright (c)  3.2020
 * This file (config.NetworkConfig) is part of NeuralNetwork.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Screamer  <999screamer999@gmail.com>
 */

import java.util.ArrayList;
import java.util.List;

public class NetworkConfig {
	public static Function SIGMOID = (s -> 1 / (1 + Math.exp(-s)));
	private List<Integer> neuronsOnLayer;
	private Integer inputArraySize;
	private Function func;

	public NetworkConfig() {
		this(0, new ArrayList<>(), NetworkConfig.SIGMOID);
	}

	public NetworkConfig(Integer inputArraySize, List<Integer> neuronsOnLayer) {
		this(inputArraySize, neuronsOnLayer, NetworkConfig.SIGMOID);
	}

	public NetworkConfig(Integer inputArraySize, List<Integer> neuronsOnLayer, Function func) {
		this.neuronsOnLayer = neuronsOnLayer;
		this.inputArraySize = inputArraySize;
		this.func = func;
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
