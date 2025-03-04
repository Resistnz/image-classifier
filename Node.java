import java.util.Random;

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
                if (input > 0) return input;
                return input * LEAKY_GRADIENT;
            }

            @Override  
            public double derivative(double x)
            {
                if (x > 0) return LEAKY_GRADIENT;
                return 0;
            }
        },
        LOGISTIC
        {
            @Override
            public double activate(double input)
            {
                return 1f / (1f + Math.exp(-input));
            }

            @Override
            public double derivative(double x)
            {
                double fx = activate(x);
                return fx * (1 - fx);
            }
        };

        public abstract double activate(double x);
        public abstract double derivative(double x);
    }
    
    private NodeActivation activationType = NodeActivation.RELU;

    private Random random = new Random();

    public Node(int numInputs)
    {
        this.numInputs = numInputs;

        inputWeights = new double[numInputs];

        // Use He initialisation
        for (int i = 0; i < numInputs; i++) {
            inputWeights[i] = random.nextGaussian() * Math.sqrt(2.0 / numInputs);
        }

        bias = 0.01;

        //if (activationType == NodeActivation.LOGISTIC) System.out.println("womp womp");
        //activationType = NodeActivation.RELU;
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
        //System.out.println("Setting activation function: " + activationType);
        this.activationType = activationType;
    }

    private double ActivationFunction(double input)
    {
        return activationType.activate(input);
    }

    public NodeActivation GetNodeActivation()
    {
        return activationType;
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