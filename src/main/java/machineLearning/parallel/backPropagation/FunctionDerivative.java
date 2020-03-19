/*
 * Copyright (c)  3.2020
 * This file (FunctionDerivative) is part of NeuralNetwork.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Screamer  <999screamer999@gmail.com>
 */

package machineLearning.parallel.backPropagation;

import neuralNetwork.config.Function;
import neuralNetwork.config.Functions;

public interface FunctionDerivative {

	static double functionDerivative(Function func, double sum) {
		Function derivative = Functions.DERIVATIVES.getOrDefault(func, s -> s);
		return derivative.calculate(sum);
	}
}
