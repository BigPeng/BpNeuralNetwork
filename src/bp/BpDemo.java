package bp;

import data.Dataset;

public class BpDemo {
	/**
	 * 对图像数字进行训练
	 */
	public static void trainDitalClassifier() {
		String fileName = "sample-data/image_data.txt";
		Dataset dataset = Dataset.load(fileName, ",", 1024, false);
		// 用于分类，对输出值进行01编码
		BPNetwork bp = new BPNetwork(new int[] { 1024, 100, 4 }, true);
		// 设置类标
		bp.setLables("sample-data/idcard.index");
		bp.trainModel(dataset, 0.999);
		bp.saveModel("model/model1438.model");
		ConcurenceRunner.stop();
	}

	/**
	 * 使用bp对ln(x1^1+x2^2+1)/ln(3)进行拟合
	 */
	public static void logFunc() {
		String fileName = "sample-data/func_train.txt";
		Dataset dataset = Dataset.load(fileName, "\t", 2, false);
		// 对函数进行拟合，不用对输出值进行编码
		BPNetwork bp = new BPNetwork(new int[] { 2, 20, 1 }, false);
		bp.trainModel(dataset, 1);
		dataset = null;
		String testName = "sample-data/func_test.txt";
		Dataset testset = Dataset.load(testName, "\t", 2, false);
		String outName = "sample-data/func_test.predict";
		bp.predict(testset, outName);
		ConcurenceRunner.stop();
	}
	
	public static void main(String[] args) {
		logFunc();
		//trainDitalClassifier();
	}
}
