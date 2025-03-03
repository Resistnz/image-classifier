import java.util.Arrays;

public class NeuralNet {
    public static void main(String[] args) 
    {
        /*double[] inputs = new double[4096];
        
        try
        {
            inputs = Image.ImageToArray("zebra-ii-square.jpg");
        }
        catch (IOException e) { e.printStackTrace(); }*/
        
        double[] inputs = new double[]{0.3, 0.1, 0.5, 0.4, 0.5, 0.9, 0.2, 0.3, 0.2, 0.7};

        double[] targets = new double[]{0};
        int[] layerSizes = new int[]{10, 5, 1};
        int trainingIterations = 2;
        double maxError = 0; // Finish iterating once the error is less than this
        double learningRate = 0.5;

        // Create the layers
        Layer[] layers = CreateLayers(layerSizes, inputs);

        System.out.print("\n");

        for (int i = 0; i < trainingIterations; i++) 
        {
            //int progressPercent = (int)(100 * i / trainingIterations);
            //System.out.print("\rCalculating iteration " + i + "/" + trainingIterations + " (" + progressPercent + "%)");
            
            // Forward propagate
            for (Layer layer : layers)
            {
                layer.Calculate();
            }

            // Calculate error
            Backpropagate(layers, targets, learningRate);

            double totalError = layers[layerSizes.length - 1].CalculateTotalError(targets);

            if (totalError < maxError) 
            {
                trainingIterations = i;
                break;
            }
            
            System.out.println("Squared error: " + totalError);

            double[] outputs = new double[targets.length];
            for (int j = 0; j < outputs.length; j++)
            {
                outputs[j] = layers[layerSizes.length - 1].nodes[j].activation;
            }

            System.out.println("Actual output: " + Arrays.toString(outputs));
        }

        double totalError = layers[layerSizes.length - 1].CalculateTotalError(targets);

        System.out.println("Training ran for " + trainingIterations + " iterations with a learning rate of " + learningRate);
        System.out.println("\nTraining inputs are " + Arrays.toString(inputs));
        System.out.println("\nTarget output: " + Arrays.toString(targets));

        // Get the outputs
        double[] outputs = new double[targets.length];
        for (int i = 0; i < outputs.length; i++)
        {
            outputs[i] = layers[layerSizes.length - 1].nodes[i].activation;
        }

        System.out.println("Actual output: " + Arrays.toString(outputs));

        System.out.println("Squared error: " + totalError);
    }

    // Create layers based on the sizes provided and inputs
    public static Layer[] CreateLayers(int[] layerSizes, double[] inputs)
    {
        int numLayers = layerSizes.length;

        // Create the layers
        Layer[] layers = new Layer[numLayers];

        Layer inputLayer = new Layer(layerSizes[0], inputs);
        layers[0] = inputLayer;

        for (int i = 1; i < numLayers; i++) 
        {
            Layer layer = new Layer(layerSizes[i], layers[i-1], i);

            layers[i] = layer;
        }

        return layers;
    }

    public static void Backpropagate(Layer[] layers, double[] targets, double learningRate)
    {
        int numLayers = layers.length;
        
        // Assign each node its delta
        for (int layerNum = numLayers - 1; layerNum > 0; layerNum--) 
        {
            // Go through each node in this layer
            for (int i = 0; i < layers[layerNum].nodes.length; i++) 
            {
                Node node = layers[layerNum].nodes[i];
                double out = node.activation;

                // de/dout
                double errorChange = 0;

                // Simple with last layer
                if (layerNum == numLayers - 1)
                {
                    double target = targets[i];
                    errorChange = out - target;
                }
                // Otherwise more complicated
                else
                {
                    // Sum the deltas of the nodes in the next layer
                    Layer nextLayer = layers[layerNum + 1];

                    for (Node nextNode : nextLayer.nodes)
                    {
                        // Multiply the delta by its connection to this node
                        errorChange += nextNode.delta * nextNode.inputWeights[i];
                    }
                }

                // Chain rule
                node.delta = errorChange * node.ActivationFunctionDerivative(node.weightedInput);
            }
        }

        // Assign weights
        for (int layerNum = numLayers - 1; layerNum > 0; layerNum--) 
        {
            // Go through each node in this layer
            for (Node node : layers[layerNum].nodes) 
            {            
                // Go through each weight for this node
                for (int j = 0; j < node.inputWeights.length; j++)
                {        
                    double w = node.inputWeights[j];
                    
                    // How a change in weight changes the error, de/dw
                    double weightChange = node.delta * layers[layerNum - 1].nodes[j].activation;

                    w -= learningRate * weightChange;

                    //System.out.println("Changed weight from " + node.inputWeights[j] +  " to " + w);
                    //System.out.println("Changed bias from " + node.bias +  " to " + (node.bias-node.delta));

                    // Update the weight
                    node.inputWeights[j] = w;   
                    node.bias -= learningRate * node.delta; // Bias is essentially a node with activation 1
                }
            }
        }
    }
}