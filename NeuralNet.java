import java.util.Arrays;
import java.io.IOException;

public class NeuralNet {
    public static void main(String[] args) 
    {
        int numImages = 2;
        double[][] inputs = new double[numImages][4096];
        
        try
        {
            for (int i = 0; i < numImages; i++)
            {
                inputs[i] = Image.ImageToArray("zebras/" + i + ".jpg");
            }
        }
        catch (IOException e) { e.printStackTrace(); }

        // Parameters
        double[] targets = new double[]{1};
        int[] layerSizes = new int[]{4096, 2000, 1};
        int trainingIterations = 10;
        double learningRate = 0.01;  

        // Create the layers
        Layer[] layers = CreateLayers(layerSizes, inputs[0]);

        // Perform each iteration
        for (int i = 0; i < trainingIterations; i++) 
        {
            // Loop over each image
            for (int j = 0; j < numImages; j++)
            {
                // Forward propagate
                for (Layer layer : layers)
                {
                    layer.Calculate();
                }

                // Calculate error
                Backpropagate(layers, targets, learningRate);
            }
        }

        System.out.println("\nTarget output: " + Arrays.toString(targets));

        // Get the outputs
        double[] outputs = new double[targets.length];
        for (int i = 0; i < outputs.length; i++)
        {
            outputs[i] = layers[layerSizes.length - 1].nodes[i].activation;
        }

        System.out.println("Actual output: " + Arrays.toString(outputs));
        System.out.println("Error: " + layers[layerSizes.length - 1].CalculateTotalError(targets));
    }

    public static Layer[] CreateLayers(int[] layerSizes, double[] inputs)
    {
        int numLayers = layerSizes.length;

        // Create the layers
        Layer[] layers = new Layer[numLayers];

        Layer inputLayer = new Layer(layerSizes[0], inputs);
        layers[0] = inputLayer;

        for (int i = 1; i < numLayers; i++) 
        {
            // Use logistic only for last layer
            var activationType = Node.NodeActivation.RELU;
            if (i == numLayers - 1) activationType = Node.NodeActivation.LOGISTIC;
            
            Layer layer = new Layer(layerSizes[i], layers[i-1], i, activationType);

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
                    
                    // Node delta for last layer
                    errorChange = -(target/out - (1-target)*(1-out));
                }
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
                    // How a change in weight changes the error, de/dw
                    double weightChange = node.delta * layers[layerNum - 1].nodes[j].activation;

                    // Update the weight
                    node.inputWeights[j] -= learningRate * weightChange;
                    node.bias -= learningRate * node.delta; // Bias is essentially a node with activation 1
                }
            }
        }
    }
}