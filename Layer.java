
public class Layer
{
    public Node[] nodes;
    private final Layer previousLayer;

    public int numNodes;
    public int layerID;
    private int previousLayerNodes = 0;

    public Layer(int numNodes, Layer previousLayer, int layerID, Node.NodeActivation activationType)
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

            node.SetActivationFunction(activationType);

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

            nodes[i] = node;
        }
    }

    public void Calculate()
    {
        //long lastTimestamp = System.nanoTime();
        //long startTime = lastTimestamp;

        //long a = 0;
        //long b = 0;
        //long c = 0;
        //int itertracker = 0;
        
        if (previousLayer == null) return;

        // Get all inputs from previous layer
        int numNodesInPreviousLayer = previousLayerNodes;
        double inputs[] = new double[numNodesInPreviousLayer];

        int count = 0;
        
        // Calculate every node
        for (int i = 0; i < numNodes; i++)
        {
            Node node = nodes[i];

            //c += System.nanoTime() - lastTimestamp;
            //lastTimestamp = System.nanoTime();

            count = 0;
            for (int j = 0; j < previousLayerNodes; j++)
            {
                inputs[count] = previousLayer.nodes[j].activation;
                count++;
            }
 
            //a += System.nanoTime() - lastTimestamp;
            //lastTimestamp = System.nanoTime();

            // Calculate this node
            node.CalculateOutput(inputs);

            //b += System.nanoTime() - lastTimestamp;
            //lastTimestamp = System.nanoTime();
        }

        //System.out.println("Creating arrays took " + (c) / 1e6 + " ms");
        //System.out.println("Getting previous inputs took " + (a) / 1e6 + " ms and used " + itertracker + " iterations");
        //System.out.println("Node calculating took " + (b) / 1e6 + " ms");
        
        //System.out.println("Layer calculation took " + (System.nanoTime() - startTime) / 1e6 + " ms\n");
        //lastTimestamp = System.nanoTime();
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