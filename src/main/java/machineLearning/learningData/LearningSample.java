/*
 * Copyright (c)  3.2020
 * This file (LearningSample) is part of NeuralNetwork.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Screamer  <999screamer999@gmail.com>
 */

package machineLearning.learningData;

public class LearningSample {
	private double[] input;
	private double[] ideal;

	public LearningSample(double[] input, double[] ideal) {
		this.input = input;
		this.ideal = ideal;
	}

	public double[] getInput() {
		return input;
	}

	public void setInput(double[] input) {
		this.input = input;
	}

	public double[] getIdeal() {
		return ideal;
	}

	public void setIdeal(double[] ideal) {
		this.ideal = ideal;
	}

}
