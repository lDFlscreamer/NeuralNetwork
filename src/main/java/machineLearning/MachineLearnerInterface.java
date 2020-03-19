/*
 * Copyright (c)  3.2020
 * This file (MachineLearninterface) is part of NeuralNetwork.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Screamer  <999screamer999@gmail.com>
 */

package machineLearning;

import machineLearning.learningData.LearningSample;

import java.util.List;

public interface MachineLearnerInterface {

	void learnNetwork(List<LearningSample> data);
}
