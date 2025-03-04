public class Node
{   
    private final int numInputs;

    public double[] inputWeights;
    public double bias;
    public double delta;

    public double activation;
    public double weightedInput;

    public boolean isInputNode;

    private final static double LEAKY_GRADIENT = 0.01;

    public static enum NodeActivation
    {
        RELU
        {
            @Override
            public double activate(double input)
            {
                //System.out.println("Input: " + input + " Output: " + input * LEAKY_GRADIENT);
                if (input < 0) return input * LEAKY_GRADIENT;
                return input;
            }

            @Override  
            public double derivative(double x)
            {
                //System.out.println("Input: " + x + " Derivative Output: " + (x > 0 ? 1 : LEAKY_GRADIENT));
                if (x > 0) return 1;
                return LEAKY_GRADIENT;
            }
        },
        LOGISTIC
        {
            @Override
            public double activate(double input)
            {
                //System.out.println("Input: " + input + " Output: " + (1f / (1f + Math.exp(-input))));
                return 1f / (1f + Math.exp(-input));
            }

            @Override
            public double derivative(double x)
            {
                double fx = activate(x);
                //System.out.println("Input: " + x + " Derivative Output: " + (fx * (1 - fx)));
                return fx * (1 - fx);
            }
        };

        public abstract double activate(double x);
        public abstract double derivative(double x);
    }
    
    private NodeActivation activationType;

    public Node(int numInputs)
    {
        this.numInputs = numInputs;

        inputWeights = new double[numInputs];

        // Fill with random numbers
        for (int i = 0; i < numInputs; i++) {
            inputWeights[i] = Math.random();
        }

        bias = 0;

        activationType = NodeActivation.RELU;
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

    public void SetActivationFunction(NodeActivation activationType)
    {
        this.activationType = activationType;
    }

    private double ActivationFunction(double input)
    {
        return activationType.activate(input);
    }

    public double ActivationFunctionDerivative(double x)
    {
        return activationType.derivative(x);
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

        activation = ActivationFunction(weightedInput + bias);
    }
}