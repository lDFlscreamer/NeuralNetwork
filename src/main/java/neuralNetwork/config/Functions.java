/*
 * Copyright (c)  3.2020
 * This file (Functions) is part of NeuralNetwork.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Screamer  <999screamer999@gmail.com>
 */

package neuralNetwork.config;

import java.util.HashMap;

public class Functions {
	public static Function SIGMOID = (s -> 1 / (1 + Math.exp(-s)));

	public static HashMap<Function,Function> DERIVATIVES=new HashMap<>();
	static {
		DERIVATIVES.put(SIGMOID,(s -> s * (1 - s)));
	}

}
