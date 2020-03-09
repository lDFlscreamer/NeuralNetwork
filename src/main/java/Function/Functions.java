package Function;

public enum Functions {

	SIGMOID(s-> 1/(1+Math.exp(-s))),
	;

	private final Function function;

	Functions(Function func) {
		this.function =func;
	}

}
