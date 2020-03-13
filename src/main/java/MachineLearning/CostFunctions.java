/*
 * Copyright (c)  3.2020
 * This file (CostFunctions) is part of NeuralNetwork.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Screamer  <999screamer999@gmail.com>
 */

package MachineLearning;

public class CostFunctions {
	public static final Cost COST = (s, s1) -> Math.pow(s1 - s, 2);

}
