package bp;
import data.Dataset;

public abstract class NeuralNetwork {

	public abstract void trainModel(Dataset trainSet,double threshold);
	public abstract void predict(Dataset testSet,String outName);
	

}
