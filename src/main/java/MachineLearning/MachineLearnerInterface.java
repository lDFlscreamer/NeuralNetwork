/*
 * Copyright (c)  3.2020
 * This file (MachineLearnerInterface) is part of NeuralNetwork.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Screamer  <999screamer999@gmail.com>
 */

package MachineLearning;

import MachineLearning.LearningData.LearningSample;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

public interface MachineLearnerInterface {
	void computeAdjust(double[] input, double[] ideal, Map<Integer, Double> biasAdjust, Map<Integer, Map<Integer, Double>> weightAdjust) throws ExecutionException, InterruptedException;

	void learnNetwork(List<LearningSample> data);
}