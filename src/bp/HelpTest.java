package bp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.swing.text.html.parser.DTD;

import data.Dataset;
import data.Log;
import data.Test;

public class HelpTest {
	public static void testArray() {
		List<Double> data = new ArrayList<Double>(4);
		data.set(0, 0.3);
		data.add(0.0);
		data.add(1.0);
		data.add(2.0);
		data.add(3.0);
		data.add(3.0);
		System.out.println(data);
	}

	public static void testArrayOrder() {
		int m = 10000;
		int n = 1000;
		long start = System.currentTimeMillis();
		double[][] data = new double[m][n];
		for (int j = 0; j < n; j++)
			for (int i = 0; i < m; i++)
				data[i][j] = Math.sqrt(i * Math.PI % j * i) / Math.E;
		long end = System.currentTimeMillis();
		System.out.println(end - start);
	}

	public static void testBinary(int n) {
		Double lable = 7.0;
		String binary = Integer.toBinaryString(lable.intValue());
		System.out.println(binary);
		System.out.println(Arrays.toString(binary.getBytes()));
		byte[] bytes = binary.getBytes();
		int[] encode = new int[n];
		for (int i = 0; i < n; i++) {
			if (i < bytes.length)
				encode[i] = bytes[i] - '0';
			else
				encode[i] = 0;
		}
		System.out.println(Arrays.toString(encode));
	}
	public static void testBP(){
		String fileName = "data/poker-hand-training-order.txt";
		Dataset dataset = Dataset.load(fileName, ",", 10);
		BPNetwork bp = new BPNetwork(new int[]{10,80,4});
		bp.trainModel(dataset,0.98);
//		String testName = "data/poker-hand-test-order.nolable";
//		Dataset testset = Dataset.load(testName, ",", -1);
//		String outName = "data/test.predict";
//		bp.predict(testset,outName);
		
		String testName = "data/poker-hand-test-order.csv";
		Dataset testset = Dataset.load(testName, ",", 10);
		double p = bp.test(testset);
		
	}
	
	
	public static void testHandWirte(){
		String fileName = "data/train.format";
		Dataset dataset = Dataset.load(fileName, ",", 784);
		BPNetwork bp = new BPNetwork(new int[]{784,100,4});
		bp.trainModel(dataset,0.99);
		dataset= null;
		String testName = "data/test.format";
		Dataset testset = Dataset.load(testName, ",", -1);
		String outName = "data/test.predict";
		bp.predict(testset,outName);
	}
	
	public static void testAotuEncode(){
		String fileName = "data/train-hw.split";
		Dataset dataset = Dataset.load(fileName, ",", 784);
		BPNetwork bp = new BPNetwork(new int[]{784,20,4});
		bp.trainModel(dataset,0.995);
		dataset= null;
		String testName = "data/test-hw.split";
		Dataset testset = Dataset.load(testName, ",", 784);	
		bp.test(testset);
	}
	public static void testTest(){
		Test t = new Test();
		double[] d = t.getData();
		for(int i=0;i<d.length;i++)
			d[i] = i*i;
		t.print();
	}
	
	public static void testTinyNet(){
		String modelName = "data/tiny.model";
		BPNetwork bp = BPNetwork.loadModel(modelName);
		String fileName = "data/tiny.txt";
		Dataset dataset = Dataset.load(fileName, ",", 3);
		bp.trainModel(dataset,0.98);
		
	}
	/**
	 * ¶Á×Ö¿â½øÐÐÑµÁ·
	 */
	public static void testWords(){
		String fileName = "data/3500.db";
		Dataset dataset = Dataset.load(fileName, ",", 1024);
		BPNetwork bp = new BPNetwork(new int[]{1024,150,12});	
		//BPNetwork bp = BPNetwork.loadModel("model/testWords.model");
		bp.trainModel(dataset,0.85);
//		String testName = "data/3510_make.txt";
//		Dataset testset = Dataset.load(testName, ",", 1024);	
//		bp.test(testset);
//		bp.saveModel("model/testWords.model");
	}
	
	public static void testMove(){
		int lable = 14;
		System.out.println((1<<(4-1)));
		lable = lable - (1<<(4-1));
		System.out.println(lable);
	}
	
	/**
	 * ²âÊÔ¿¨ºÅµÈÊý×Ö
	 */
	public static void testIdCarddDital(){
		String fileName = "data/merge.shuffle";
		Dataset dataset = Dataset.load(fileName, ",", 1024);
		BPNetwork bp = new BPNetwork(new int[]{1024,100,4});	
		bp.trainModel(dataset,0.9999);
		String testName = "data/idcard_digit.xiong.shuffle.all";
		Dataset testset = Dataset.load(testName, ",", 1024);	
		bp.test(testset);
		bp.saveModel("model/model1438.model");
		ConcurenceRunner.stop();
	}
	public static void main(String[] args) {
		// testArray();
		// testArrayOrder();
		//testBinary(5);
		//testBP();
		//testTest();
		//testTinyNet();
		//testHandWirte();
		//testAotuEncode();
		//testMove();
		//testWords();
		testIdCarddDital();
	}

}
