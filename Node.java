
import Node.ActivationType;

public class Node
{   
    private final int numInputs;

    public double[] inputWeights;
    public double bias;
    public double delta;

    public double activation;
    public double weightedInput;

    public boolean isInputNode;

    private final static double LEAKY_GRADIENT = 0.3;

    public static enum ActivationType
    {
        RELU
        {
            @Override
            public double activate(double input)
            {
                if (input < 0) return input * LEAKY_GRADIENT;
                return input;
            }

            @Override  
            public double derivative(double x)
            {
                if (x > 0) return 1;
                return LEAKY_GRADIENT;
            }
        },
        LOGISTIC
        {
            @Override
            public double activate(double input)
            {
                if (input < 0) return input * LEAKY_GRADIENT;
                return input;
            }

            @Override
            public double derivative(double x)
            {
                if (x > 0) return 1;
                return LEAKY_GRADIENT;
            }
        };

        public abstract double activate(double x);
        public abstract double derivative(double x);
    }
    
    private ActivationType activationType;

    public Node(int numInputs)
    {
        this.numInputs = numInputs;

        inputWeights = new double[numInputs];

        // Fill with random numbers
        for (int i = 0; i < numInputs; i++) {
            inputWeights[i] = Math.random();
        }

        bias = 0;

        activationType = ActivationType.RELU;
    }

    // Add custom weights and bias
    public Node(int numInputs, double[] weights, double bias)
    {
        this(numInputs);

        inputWeights = weights;

        this.bias = bias;

        activationType = ActivationType.RELU;
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
        if (input < 0) return input * LEAKY_GRADIENT;
        
        return input;
    }

    private static double reLUDerivative(double x)
    {
        if (x > 0) return 1;
        return LEAKY_GRADIENT;
    }

    private static double logistic(double input)
    {
        double result = 1f / (1f + Math.exp(-input));

        return result;
    }

    private static double logisticDerivative(double x)
    {
        return x * (1 - x);
    }

    public double activationFunction(double input)
    {
        double activated = 0;
        
        switch (activationType)
        {
            case ActivationType.RELU:
                activated = reLU(input);
                break;
            case ActivationType.LOGISTIC:
                activated = logistic(input);
                break;
        }

        return activated;
    }

    //public static 
    
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