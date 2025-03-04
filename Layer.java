
public class Layer
{
    public Node[] nodes;
    private final Layer previousLayer;

    public int numNodes;
    public int layerID;
    private int previousLayerNodes = 0;

    public Layer(int numNodes, Layer previousLayer, int layerID)
    {
        this.layerID = layerID;
        
        this.numNodes = numNodes;
        this.previousLayer = previousLayer;
        previousLayerNodes = previousLayer.nodes.length;

        nodes = new Node[numNodes];
 
        // Generate the nodes
        for (int i = 0; i < numNodes; i++) 
        {        
            Node node = new Node(previousLayerNodes);

            nodes[i] = node;
        }
    }

    // Input layer
    public Layer(int numNodes, double[] inputs)
    {
        layerID = 0;
        
        this.numNodes = numNodes;
        this.previousLayer = null;

        nodes = new Node[numNodes];
 
        // Generate the nodes
        for (int i = 0; i < numNodes; i++) 
        {        
            Node node = new Node(true, inputs[i]);
            
            // Use logistic for last node
            if (i == numNodes - 1) node.SetActivationFunction(Node.NodeActivation.LOGISTIC);

            nodes[i] = node;
        }
    }

    public void Calculate()
    {
        if (previousLayer == null) return;
        
        // Calculate every node
        for (int i = 0; i < numNodes; i++)
        {
            Node node = nodes[i];

            // Get all inputs from previous layer
            int numNodesInPreviousLayer = previousLayerNodes;
            double inputs[] = new double[numNodesInPreviousLayer];

            int count = 0;
            for (Node previousNode : previousLayer.nodes)
            {
                inputs[count] = previousNode.activation;
                count++;
            }
 
            // Calculate this node
            node.CalculateOutput(inputs);
        }
    }

    // Squared error function (multiplied by 0.5)
    public double CalculateTotalError(double[] target)
    {
        double outputSum = 0;

        for (int i = 0; i < target.length; i++) 
        {
            double difference = (target[i] - nodes[i].activation);
            difference *= difference;

            outputSum += difference;
        }

        return outputSum * 0.5;
    }
}