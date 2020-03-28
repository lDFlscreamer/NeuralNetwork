/*
 * Copyright (c)  3.2020
 * This file (BackPropagationInterface) is part of NeuralNetwork.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Screamer  <999screamer999@gmail.com>
 */

package machineLearning.serial.backPropagation;

import neuralNetwork.Layer;
import neuralNetwork.config.Function;

public interface BackPropagationSerialInterface {
	double functionDerivative(Function func, double sum);

	double dCost_dSignal(Layer current, Integer indexOfNeuron, double[] ideal);
}
