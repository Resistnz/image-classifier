public class Node
{   
    private final int numInputs;

    public double[] inputWeights;
    public double bias;
    public double delta;

    public double activation;
    public double weightedInput;

    public boolean isInputNode;

    private final static double leakyGradient = 0.01;

    public Node(int numInputs)
    {
        this.numInputs = numInputs;

        inputWeights = new double[numInputs];

        // Fill with random numbers
        for (int i = 0; i < numInputs; i++) {
            inputWeights[i] = Math.random();
        }

        bias = 0;
    }

    // Add custom weights and bias
    public Node(int numInputs, double[] weights, double bias)
    {
        this(numInputs);

        inputWeights = weights;

        this.bias = bias;
    }

    // Handle input nodes
    // An input node has no weights and simply outputs a number
    public Node(boolean isInputNode, double activation)
    {
        this(0);
        this.isInputNode = isInputNode;
        this.activation = activation;
    }

    private static double reLU(double input)
    {
        if (input < 0) return input * leakyGradient;
        
        return input;
    }

    public static double reLUDerivative(double x)
    {
        if (x > 0) return 1;
        return leakyGradient;
    }

    private static double logistic(double input)
    {
        double result = 1f / (1f + Math.exp(-input));

        return result;
    }

    public static double logisticDerivative(double x)
    {
        return x * (1 - x);
    }
    
    public void CalculateOutput(double[] inputs)
    {
        // Don't calculate input nodes
        if (isInputNode) return;

        weightedInput = 0;

        for (int i = 0; i < numInputs; i++)
        {
            weightedInput += inputs[i] * inputWeights[i];
        }

        activation = reLU(weightedInput + bias);
    }
}