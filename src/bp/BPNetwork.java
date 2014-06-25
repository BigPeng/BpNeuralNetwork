package bp;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;

import data.Dataset;
import data.Dataset.Record;
import data.Log;
import data.Util;

public class BPNetwork extends NeuralNetwork {
	private static final double LEARN_RATE = 0.1;
	// 神经网络的层数,包括输入层\输出层\隐藏层
	private int layerNum;
	// 每一层神经网络
	private List<NeuralLayer> layers;
	// 每一层神经单元的个数
	private int[] layerSize;
	// 并发调度器
	ConcurenceRunner runner;
	// 打开自编码
	private final boolean autoEncodeOn = false;
	private BPNetwork autoNetwork = null;
	private static AtomicBoolean stopTrain = new AtomicBoolean(false);

	private int maxLable = -1;
	private List<String> lables;
	// cnn的宽度
	private static final int WIDTH = 10;

	public BPNetwork(int[] layerSize) {
		this.layerNum = layerSize.length;
		this.layers = new ArrayList<NeuralLayer>(layerNum);
		this.layerSize = layerSize;
		runner = new ConcurenceRunner();
	}

	public BPNetwork() {
		this.layers = new ArrayList<NeuralLayer>();
		runner = new ConcurenceRunner();
	}

	/**
	 * 训练模型
	 */
	@Override
	public void trainModel(Dataset trainSet, double threshold) {
		if (autoEncodeOn) {// 开启自编码
			try {
				int featureSize = 300;
				autoNetwork = autoEncode(trainSet, featureSize);// 自编码特征提取
				trainSet = autoEncodeDataset(autoNetwork, trainSet, featureSize);
				layerSize[0] = featureSize;
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		// 初始化神经层
		initLayer();
		new Lisenter().start();// 开启停止监听
		double precison = test(trainSet, false);
		// double precison = 0.1;
		while (precison < threshold && !stopTrain.get()) {
			train(trainSet, false);
			precison = test(trainSet);
			// break;
		}
		// runner.stop();
		// 准确率

	}

	public void setMaxLable(int maxLable) {
		this.maxLable = maxLable;
	}

	private void initLayer() {
		for (int i = 0; i < layerSize.length; i++) {
			NeuralLayer layer = new NeuralLayer(i, layerSize[i],
					i > 0 ? layerSize[i - 1] : 0);
			layers.add(layer);
		}
	}

	/**
	 * 预测数据
	 */
	@Override
	public void predict(Dataset testSet, String outName) {
		try {
			int max = maxLable;
			boolean hasLable = testSet.getLableIndex() != -1 ? true : false;
			PrintWriter out = new PrintWriter(new File(outName));
			Iterator<Record> iter = testSet.iter();
			int rightCount = 0;
			while (iter.hasNext()) {
				Record record = iter.next();
				// 设置输入层的输出值为当前记录值
				getLayer(0).setOutputs(record);
				// 更新后面各层的输出值
				updateOutput();
				double[] output = getLayer(layerNum - 1).getOutputs();
				int lable = Util.binaryArray2int(output);
				if (lable > max)
					lable = lable - (1 << (output.length - 1));
				out.write(lable + "\n");
				if (hasLable && isSame(output, record, false)) {
					rightCount++;
				}
			}
			out.flush();
			out.close();
			Log.i("precision", "" + (rightCount * 1.0 / testSet.size()));
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * 预测数据
	 */

	public List<Integer> predict(Dataset testSet) {
		try {
			List<Integer> result = new ArrayList<Integer>();
			int max = maxLable;
			boolean hasLable = testSet.getLableIndex() != -1 ? true : false;
			Iterator<Record> iter = testSet.iter();
			int rightCount = 0;
			while (iter.hasNext()) {
				Record record = iter.next();
				// 设置输入层的输出值为当前记录值
				getLayer(0).setOutputs(record);
				// 更新后面各层的输出值
				updateOutput();
				double[] output = getLayer(layerNum - 1).getOutputs();
				int lable = Util.binaryArray2int(output);
				if (lable > max)
					lable = lable - (1 << (output.length - 1));
				if (hasLable && isSame(output, record, false)) {
					rightCount++;
				}
				result.add(lable);
			}
			Log.i("precision", "" + (rightCount * 1.0 / testSet.size()));
			return result;
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}

	/***
	 * 根据数据,自编码进行特征提取,实质上是一个三层神经网络,输入层为原始数据,
	 * 输出层是对原始数据的拟合
	 * 
	 * 
	 * @param dataSet
	 * @param size
	 *            神经单元个数
	 * @return 返回自编码的神经层,维度一般比输入层低,达到降维和特征提取的效果
	 * @throws InterruptedException
	 */
	private BPNetwork autoEncode(Dataset dataSet, int size)
			throws InterruptedException {
		Log.i("autoEncode ", "size:" + size);
		int inputLayerSize = layerSize[0];
		// 自编码网络,输入层个数与输出层个数相等,隐藏层个数较少
		int[] layerSize = { inputLayerSize, size, inputLayerSize };
		BPNetwork aotuNN = new BPNetwork(layerSize);
		aotuNN.initLayer();
		double precison = 0.1;
		while (precison < 0.8) {
			aotuNN.train(dataSet, true);
			precison = aotuNN.test(dataSet, true);
			// break;
		}
		aotuNN.layers.remove(2);// 最后一层已经不需要了
		aotuNN.layerNum = 2;
		aotuNN.layerSize = Arrays.copyOfRange(aotuNN.layerSize, 0, 2);
		return aotuNN;

	}

	private Dataset autoEncodeDataset(BPNetwork aotuNN, Dataset dataset,
			int size) throws InterruptedException {
		// 自编码中层有多少个结点，特征个数就有多少个,类标放在最后
		Dataset featureSet = new Dataset(size);
		Iterator<Record> iter = dataset.iter();
		while (iter.hasNext()) {
			Record record = iter.next();
			aotuNN.getLayer(0).setOutputs(record);
			aotuNN.updateOutput();
			double[] features = aotuNN.getLayer(1).getOutputs();
			featureSet.append(features, record.getLable());
		}
		dataset.clear();
		Log.i("autoEncode ", "complete");
		// 第二层即为中间层
		return featureSet;
	}

	/**
	 * 训练一个神经网络,输入层已经给出
	 * 
	 * @param inputLayer
	 * @param layerNum
	 * @return
	 */
	private void train(Dataset dataSet, boolean isAutoEncode) {
		Iterator<Record> iter = dataSet.iter();
		while (iter.hasNext()) {
			try {
				train(iter.next(), isAutoEncode);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}

	public double test(Dataset dataSet) {
		return test(dataSet, false);
	}

	static class Lisenter extends Thread {

		@Override
		public void run() {
			System.out.println("输入&结束迭代");
			while (true) {
				try {
					int a = System.in.read();
					if (a == '&') {
						stopTrain.compareAndSet(false, true);
						break;
					}
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			System.out.println("Lisenter stop");
		}

	}

	/**
	 * 测试准确率
	 * 
	 * @param dataSet
	 * @param autoEncode
	 */
	private double test(Dataset dataSet, boolean autoEncode) {
		helpCount = 0;
		// Log.i("errors ",
		// Arrays.toString(getLayer(2).getErrors()));
		int rightCount = 0;
		try {
			if (autoEncodeOn && autoNetwork != null) {
				dataSet = autoEncodeDataset(autoNetwork, dataSet, autoNetwork
						.getLayer(1).getSize());
			}
			Iterator<Record> iter = dataSet.iter();
			while (iter.hasNext()) {

				Record record = iter.next();
				// 设置输入层的输出值为当前记录值
				getLayer(0).setOutputs(record);
				// 更新后面各层的输出值
				updateOutput();
				double[] output = getLayer(layerNum - 1).getOutputs();

				if (isSame(output, record, autoEncode)) {
					rightCount++;
				}

			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		double p = rightCount * 1.0 / dataSet.size();
		Log.i(rightCount + "", p + "");
		return p;
	}

	private int helpCount = 0;

	private boolean isSame(double[] output, Record record, boolean autoEncode) {
		double[] target;
		if (autoEncode)// 自编码时计算正确率是用属性值
			target = record.getAttrs();
		else
			// 训练模型时，自编码时用类标
			target = record.getDoubleEncodeTarget(output.length);
		boolean r = true;
		for (int i = 0; i < output.length; i++)
			if (Math.abs(output[i] - target[i]) > 0.5) {
				r = false;
				break;
			}
		if (helpCount++ % 3000 == -1)
			Log.i("isSame", record.getLable() + " " + Arrays.toString(output)
					+ " " + Arrays.toString(target) + " " + r);

		return r;
	}

	/***
	 * 使用一条记录训练网络
	 * 
	 * @param record
	 * @throws InterruptedException
	 */
	private void train(Record record, boolean isAutoEncode)
			throws InterruptedException {
		// 设置输入层的输出值为当前记录值
		getLayer(0).setOutputs(record);
		// 更新后面各层的输出值
		updateOutput();
		// 更新输出层的错误率
		updateOutputLayerErrors(record, isAutoEncode);
		// 更新每一层的错误率
		updateErrors();
		// 更新权重
		updateWeights();
		// 更新偏置
		updateBiases();

	}

	/**
	 * 更新偏置
	 * 
	 * @throws InterruptedException
	 */
	private void updateBiases() throws InterruptedException {
		for (int layerIndex = 1; layerIndex < layerNum; layerIndex++) {
			NeuralLayer layer = getLayer(layerIndex);
			final double[] biases = layer.getBiases();
			final double[] errors = layer.getErrors();
			int cpuNum = ConcurenceRunner.cpuNum;
			cpuNum = cpuNum < errors.length ? cpuNum : 1;// 比cpu的个数小时，只用一个线程
			final CountDownLatch gate = new CountDownLatch(cpuNum);
			int fregLength = errors.length / cpuNum;
			for (int cpu = 0; cpu < cpuNum; cpu++) {
				int start = cpu * fregLength;
				int tmp = (cpu + 1) * fregLength;
				int end = tmp <= errors.length ? tmp : errors.length;
				Task task = new Task(start, end) {
					@Override
					public void process(int start, int end) {
						for (int j = start; j < end; j++) {
							biases[j] += LEARN_RATE * errors[j];
						}
						gate.countDown();
					}
				};
				runner.run(task);
			}
			gate.await();
		}

	}

	/**
	 * 利用记录的类标，反馈更新输出值得错误率
	 * 
	 * @param record
	 * @param isAutoEncode
	 * @throws InterruptedException
	 */
	private void updateOutputLayerErrors(Record record, boolean isAutoEncode)
			throws InterruptedException {
		NeuralLayer outputLayer = getLayer(layerNum - 1);
		double[] tar;
		if (isAutoEncode)// 自编码时是用属性值来计算误差
			tar = record.getAttrs();
		else
			// 训练网络时使用类标来计算误差
			tar = record.getDoubleEncodeTarget(layerSize[layerNum - 1]);
		final double[] target = tar;
		final double[] errors = outputLayer.getErrors();
		final double[] outputs = outputLayer.getOutputs();
		int cpuNum = ConcurenceRunner.cpuNum;
		cpuNum = cpuNum < outputs.length ? cpuNum : 1;// 比cpu的个数小时，只用一个线程
		final CountDownLatch gate = new CountDownLatch(cpuNum);
		int fregLength = outputs.length / cpuNum;
		for (int cpu = 0; cpu < cpuNum; cpu++) {
			int start = cpu * fregLength;
			int tmp = (cpu + 1) * fregLength;
			int end = tmp <= outputs.length ? tmp : outputs.length;
			Task task = new Task(start, end) {

				@Override
				public void process(int start, int end) {
					for (int j = start; j < end; j++) {
						errors[j] = outputs[j] * (1 - outputs[j])
								* (target[j] - outputs[j]);

					}
					gate.countDown();
				}
			};
			runner.run(task);
		}

		gate.await();// 等待各个线程跑完

	}

	/**
	 * 更新权重
	 * 
	 * @throws InterruptedException
	 */
	private void updateWeights() throws InterruptedException {
		for (int layerIndex = 1; layerIndex < layerNum; layerIndex++) {
			NeuralLayer layer = getLayer(layerIndex);
			NeuralLayer lastLayer = getLayer(layerIndex - 1);
			final double[] lastOutputs = lastLayer.getOutputs();
			final double[][] weights = layer.getWeights();
			final double[] errors = layer.getErrors();
			// 并发运行
			int cpuNum = ConcurenceRunner.cpuNum;
			cpuNum = cpuNum < errors.length ? cpuNum : 1;// 比cpu的个数小一个时，只用一个线程
			final CountDownLatch gate = new CountDownLatch(cpuNum);
			int fregLength = errors.length / cpuNum;
			for (int cpu = 0; cpu < cpuNum; cpu++) {
				int start = cpu * fregLength;
				int tmp = (cpu + 1) * fregLength;
				int end = tmp <= errors.length ? tmp : errors.length;
				Task task = new Task(start, end) {
					@Override
					public void process(int start, int end) {
						for (int j = start; j < end; j++) {
							for (int i = 0; i < weights.length; i++) {
								weights[i][j] += (LEARN_RATE * errors[j] * lastOutputs[i]);
							}
						}
						gate.countDown();
					}

				};
				runner.run(task);
			}
			gate.await();

		}
	}

	/**
	 * 更新错误率
	 * 
	 * @throws InterruptedException
	 */
	private void updateErrors() throws InterruptedException {
		for (int i = layerNum - 2; i >= 0; i--) {
			NeuralLayer layer = getLayer(i);
			NeuralLayer nextLayer = getLayer(i + 1);
			final double[] nextErrors = nextLayer.getErrors();
			final double[] errors = layer.getErrors();
			final double[] outputs = layer.getOutputs();
			final double[][] weights = nextLayer.getWeights();
			// System.out.println("weights:"+weights.length+" next "+weights[0].length);
			// 并发运行
			int cpuNum = ConcurenceRunner.cpuNum;
			cpuNum = cpuNum < outputs.length ? cpuNum : 1;// 比cpu的个数小一个时，只用一个线程
			final CountDownLatch gate = new CountDownLatch(cpuNum);
			int fregLength = outputs.length / cpuNum;
			for (int cpu = 0; cpu < cpuNum; cpu++) {
				int start = cpu * fregLength;
				int tmp = (cpu + 1) * fregLength;
				int end = tmp <= outputs.length ? tmp : outputs.length;
				// System.out.println("start:"+start+" end:"+end+" outputs.length"+outputs.length);
				Task task = new Task(start, end) {
					@Override
					public void process(int start, int end) {
						for (int j = start; j < end; j++) {
							double tmp = 0.0;
							for (int k = 0; k < nextErrors.length; k++) {
								tmp += nextErrors[k] * weights[j][k];
							}
							errors[j] = outputs[j] * (1 - outputs[j]) * tmp;

						}
						gate.countDown();
					}
				};
				runner.run(task);
			}
			gate.await();

		}
	}

	/**
	 * 更新输出值.同时对输出层的输出进行01规整，方便输出编码
	 * 
	 * @throws InterruptedException
	 */
	private void updateOutput() throws InterruptedException {
		for (int k = 1; k < layerNum; k++) {
			NeuralLayer layer = getLayer(k);
			NeuralLayer lastLayer = getLayer(k - 1);
			final double[] lastOutputs = lastLayer.getOutputs();
			final double[] outputs = layer.getOutputs();
			final double[][] weights = layer.getWeights();
			final double[] theta = layer.getBiases();

			// 并发运行
			int cpuNum = ConcurenceRunner.cpuNum;
			cpuNum = cpuNum < outputs.length ? cpuNum : 1;// 比cpu的个数小一个时，只用一个线程
			final CountDownLatch gate = new CountDownLatch(cpuNum);
			int fregLength = outputs.length / cpuNum;
			for (int cpu = 0; cpu < cpuNum; cpu++) {
				int start = cpu * fregLength;
				int tmp = (cpu + 1) * fregLength;
				int end = tmp <= outputs.length ? tmp : outputs.length;
				final int layerIndex = k;
				final double ratio = weights.length * 1.0 / outputs.length;
				Task task = new Task(start, end) {

					@Override
					public void process(int start, int end) {
						for (int j = start; j < end; j++) {
							double tmp = 0;
							// int iStart = (int)
							// (j*ratio);
							// int iEnd = (int)
							// ((j+1)*ratio);
							// if (iEnd>
							// weights.length)
							// iEnd = weights .length;
							// for (int i = iStart; i <
							// iEnd; i++) {
							// tmp += weights[i][j] *
							// lastOutputs[i];
							// }
							for (int i = 0; i < weights.length; i++) {
								tmp += weights[i][j] * lastOutputs[i];
							}

							outputs[j] = 1 / (1 + Math.pow(Math.E,
									-(tmp + theta[j])));

							// if (layerIndex ==
							// layerNum - 1) {
							// double help =
							// outputs[j];
							//
							// // outputs[j] =
							// // outputs[j] > 0.5 ? 1
							// // : 0;// 01规整,方便编码输出
							//
							// }
						}
						gate.countDown();
					}
				};
				runner.run(task);
			}

			gate.await();// 等待所有线程跑完一轮数据

		}
	}

	/**
	 * 获取第index层的神经层，index从0开始
	 * 
	 * @param index
	 * @return
	 */
	private NeuralLayer getLayer(int index) {
		return layers.get(index);
	}

	public static BPNetwork loadModel(String fileName) {
		try {
			return loadModel(new BufferedReader(new FileReader(fileName)));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		return null;

	}

	private static BPNetwork loadModel(BufferedReader in) {
		try {
			BPNetwork bp = new BPNetwork();
			String line = in.readLine();// 第一行表示有多少层
			bp.maxLable = Integer.parseInt(line);
			line = in.readLine();
			bp.layerNum = Integer.parseInt(line);
			bp.layerSize = Util.string2ints(in.readLine());
			for (int i = 0; i < bp.layerNum; i++) {// 初始化每一层
				NeuralLayer layer = new NeuralLayer();
				layer.setSize(bp.layerSize[i]);// 设置神经单元的个数
				layer.setLayerIndex(i);// 设置层号
				line = in.readLine();
				if (!line.equals("null"))// 读偏置
					layer.setBiases(Util.string2doubles(line));
				line = in.readLine();
				if (!line.equals("null")) {// 读权重
					int[] weightSize = Util.string2ints(line);// 权重有多少行和列
					double[][] weights = new double[weightSize[0]][weightSize[1]];
					for (int j = 0; j < weightSize[0]; j++) {
						weights[j] = Util.string2doubles(in.readLine());
					}
					layer.setWeights(weights);
				}
				layer.setErrors(new double[layer.getSize()]);
				layer.setOutputs(new double[layer.getSize()]);
				bp.layers.add(layer);
			}
			in.close();
			return bp;

		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
	}

	/**
	 * 保存模型,只需要保存偏置和权重即可
	 * 
	 * @param fileName
	 */
	public void saveModel(String fileName) {
		try {
			PrintWriter out = new PrintWriter(fileName);
			out.write(maxLable + "\n");// 第一行是层数
			out.write(layerNum + "\n");// 第一行是层数
			out.write(Util.array2string(layerSize) + "\n");
			for (int i = 0; i < layerNum; i++) {// 保存每一层
				NeuralLayer layer = getLayer(i);
				double[] biases = layer.getBiases();
				if (biases == null)
					out.write("null\n");
				else {
					out.write(Util.array2string(biases) + "\n");
				}
				double[][] weights = layer.getWeights();
				if (weights == null)
					out.write("null\n");
				else {
					out.write(weights.length + ", " + weights[0].length + "\n");
					for (int k = 0; k < weights.length; k++) {
						out.write(Util.array2string(weights[k]) + "\n");
					}
				}

			}
			out.flush();
			out.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	private abstract class Task implements Runnable {
		int start, end;

		public Task(int start, int end) {
			this.start = start;
			this.end = end;
		}

		@Override
		public void run() {
			process(start, end);
		}

		public abstract void process(int start, int end);

	}

	public void setLablesFile(String fileName) {
		try {
			lables = new ArrayList<String>();
			BufferedReader reader = new BufferedReader(new FileReader(fileName));
			String line = reader.readLine();
			while (line != null) {
				lables.add(line.trim());
				line = reader.readLine();
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
