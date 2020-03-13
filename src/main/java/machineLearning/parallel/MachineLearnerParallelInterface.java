/*
 * Copyright (c)  3.2020
 * This file (MachineLearnerInterface) is part of NeuralNetwork.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Screamer  <999screamer999@gmail.com>
 */

package machineLearning.parallel;

import machineLearning.learningData.LearningSample;

import java.util.List;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinTask;

public interface MachineLearnerParallelInterface {
	List<ForkJoinTask<Void>> computeAdjust(double[] input, double[] ideal, ConcurrentMap<Integer, Double> biasAdjust, ConcurrentMap<Integer, ConcurrentMap<Integer, Double>> weightAdjust) throws ExecutionException, InterruptedException;

	void learnNetwork(List<LearningSample> data);
}