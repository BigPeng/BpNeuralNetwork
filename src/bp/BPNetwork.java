package bp;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
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
	private static final double LEARN_RATE = 0.01;
	// ������Ĳ���,���������\�����\���ز�
	private int layerNum;
	// ÿһ��������
	private List<NeuralLayer> layers;
	// ÿһ���񾭵�Ԫ�ĸ���
	private int[] layerSize;
	// ����������
	ConcurenceRunner runner;

	private static AtomicBoolean stopTrain = new AtomicBoolean(false);

	private int maxLable = -1;
	// ģ����ѵ�����ϵ����
	private List<String> lables;
	// �Ƿ��Ŀ��ֵ���б���,��1�����001��2�����010
	private boolean encodeLable;

	public BPNetwork(int[] layerSize, boolean encodeLable) {
		this.layerNum = layerSize.length;
		this.layers = new ArrayList<NeuralLayer>(layerNum);
		this.layerSize = layerSize;
		this.encodeLable = encodeLable;
		runner = new ConcurenceRunner();
	}

	public BPNetwork() {
		this.layers = new ArrayList<NeuralLayer>();
		runner = new ConcurenceRunner();
	}

	/**
	 * ѵ��ģ��
	 */
	@Override
	public void trainModel(Dataset trainSet, double threshold) {

		// ��ʼ���񾭲�
		initLayer();
		new Lisenter().start();// ����ֹͣ����
		double precison = test(trainSet);
		// double precison = 0.1;
		while (precison < threshold && !stopTrain.get()) {
			train(trainSet);
			precison = test(trainSet);
			// break;
		}
		// runner.stop();
		// ׼ȷ��

	}

	private void initLayer() {
		for (int i = 0; i < layerSize.length; i++) {
			NeuralLayer layer = new NeuralLayer(i, layerSize[i],
					i > 0 ? layerSize[i - 1] : 0);
			layers.add(layer);
		}
	}

	/**
	 * Ԥ������
	 */
	@Override
	public void predict(Dataset testSet, String outName) {
		try {

			PrintWriter out = new PrintWriter(new File(outName));
			Iterator<Record> iter = testSet.iter();
			int rightCount = 0;
			while (iter.hasNext()) {
				Record record = iter.next();
				// �������������ֵΪ��ǰ��¼ֵ
				getLayer(0).setOutputs(record);
				// ���º����������ֵ
				updateOutput();
				double[] output = getLayer(layerNum - 1).getOutputs();
				out.write(output[0] + "\n");
				/**
				 * int lable =
				 * Util.binaryArray2int(output); if
				 * (lable > max) lable = lable - (1 <<
				 * (output.length - 1));
				 * out.write(lable + "\n"); if
				 * (hasLable && isSame(output, record,
				 * false)) { rightCount++; }
				 **/
			}
			out.flush();
			out.close();
			Log.i("precision", "" + (rightCount * 1.0 / testSet.size()));
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Ԥ������
	 */

	public List<String> predict(Dataset testSet) {
		try {
			List<String> result = new ArrayList<String>();
			int max = maxLable;
			boolean hasLable = testSet.getLableIndex() != -1 ? true : false;
			Iterator<Record> iter = testSet.iter();
			int rightCount = 0;
			while (iter.hasNext()) {
				Record record = iter.next();
				// �������������ֵΪ��ǰ��¼ֵ
				getLayer(0).setOutputs(record);
				// ���º����������ֵ
				updateOutput();
				double[] output = getLayer(layerNum - 1).getOutputs();
				int lable = Util.binaryArray2int(output);
				if (lable > max)
					lable = lable - (1 << (output.length - 1));
				if (hasLable && isSame(output, record)) {
					rightCount++;
				}
				result.add(lables.get(lable));
			}
			Log.i("precision", "" + (rightCount * 1.0 / testSet.size()));
			return result;
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}

	/**
	 * ѵ��һ��������,������Ѿ�����
	 * 
	 * @param inputLayer
	 * @param layerNum
	 * @return
	 */
	private void train(Dataset dataSet) {
		Iterator<Record> iter = dataSet.iter();
		while (iter.hasNext()) {
			try {
				train(iter.next());
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}

	static class Lisenter extends Thread {

		@Override
		public void run() {
			System.out.println("����&��������");
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
	 * ����׼ȷ��
	 * 
	 * @param dataSet
	 * @param autoEncode
	 */
	public double test(Dataset dataSet) {
		helpCount = 0;
		// Log.i("errors ",
		// Arrays.toString(getLayer(2).getErrors()));
		int rightCount = 0;
		try {
			Iterator<Record> iter = dataSet.iter();
			while (iter.hasNext()) {

				Record record = iter.next();
				// �������������ֵΪ��ǰ��¼ֵ
				getLayer(0).setOutputs(record);
				// ���º����������ֵ
				updateOutput();
				double[] output = getLayer(layerNum - 1).getOutputs();

				if (isSame(output, record)) {
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

	private boolean isSame(double[] output, Record record) {
		double[] target = record.getTarget(output.length, encodeLable);
		boolean r = true;
		for (int i = 0; i < output.length; i++) {
			if (encodeLable && Math.abs(output[i] - target[i]) > 0.5) {
				r = false;
				break;
			}
			// ������ʱ��ֱ�ӱȽ���Ŀ��ֵ�Ĳ�
			if (!encodeLable && Math.abs(output[i] - target[i]) > 0.005) {
				r = false;
				break;
			}
		}
		if (helpCount++ % 30 == -1)
			Log.i("isSame", lables.get((int) (record.getLable().intValue()))
					+ " " + record.getLable() + " " + Arrays.toString(output)
					+ " " + Arrays.toString(target) + " " + r);

		return r;
	}

	/***
	 * ʹ��һ����¼ѵ������
	 * 
	 * @param record
	 * @throws InterruptedException
	 */
	private void train(Record record) throws InterruptedException {
		// �������������ֵΪ��ǰ��¼ֵ
		getLayer(0).setOutputs(record);
		// ���º����������ֵ
		updateOutput();
		// ���������Ĵ�����
		updateOutputLayerErrors(record);
		// ����ÿһ��Ĵ�����
		updateErrors();
		// ����Ȩ��
		updateWeights();
		// ����ƫ��
		updateBiases();

	}

	/**
	 * ����ƫ��
	 * 
	 * @throws InterruptedException
	 */

	private void updateBiases() throws InterruptedException {

		for (int layerIndex = 1; layerIndex < layerNum; layerIndex++) {
			NeuralLayer layer = getLayer(layerIndex);
			final double[] biases = layer.getBiases();
			final double[] errors = layer.getErrors();
			int cpuNum = ConcurenceRunner.cpuNum;
			cpuNum = cpuNum < errors.length ? cpuNum : 1;// ��cpu�ĸ���Сʱ��ֻ��һ���߳�
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
	 * ���ü�¼����꣬�����������ֵ�ô�����
	 * 
	 * @param record
	 * @param isAutoEncode
	 * @throws InterruptedException
	 */
	private void updateOutputLayerErrors(Record record)
			throws InterruptedException {
		NeuralLayer outputLayer = getLayer(layerNum - 1);

		// ѵ������ʱʹ��������������
		double[] tar = record.getTarget(layerSize[layerNum - 1], encodeLable);
		final double[] target = tar;
		final double[] errors = outputLayer.getErrors();
		final double[] outputs = outputLayer.getOutputs();
		int cpuNum = ConcurenceRunner.cpuNum;
		cpuNum = cpuNum < outputs.length ? cpuNum : 1;// ��cpu�ĸ���Сʱ��ֻ��һ���߳�
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

		gate.await();// �ȴ������߳�����

	}

	/**
	 * ����Ȩ��
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
			// ��������
			int cpuNum = ConcurenceRunner.cpuNum;
			cpuNum = cpuNum < errors.length ? cpuNum : 1;// ��cpu�ĸ���Сһ��ʱ��ֻ��һ���߳�
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
	 * ���´�����
	 * 
	 * @throws InterruptedException
	 */
	private void updateErrors() throws InterruptedException {
		for (int i = layerNum - 2; i > 0; i--) {// 2014��7��9��19:52:35
												// ��i>=0�޸�Ϊi>0��δ����
			NeuralLayer layer = getLayer(i);
			NeuralLayer nextLayer = getLayer(i + 1);
			final double[] nextErrors = nextLayer.getErrors();
			final double[] errors = layer.getErrors();
			final double[] outputs = layer.getOutputs();
			final double[][] weights = nextLayer.getWeights();
			// System.out.println("weights:"+weights.length+" next "+weights[0].length);
			// ��������
			int cpuNum = ConcurenceRunner.cpuNum;
			cpuNum = cpuNum < outputs.length ? cpuNum : 1;// ��cpu�ĸ���Сһ��ʱ��ֻ��һ���߳�
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
	 * �������ֵ.ͬʱ���������������01�����������������
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

			// ��������
			int cpuNum = ConcurenceRunner.cpuNum;
			cpuNum = cpuNum < outputs.length ? cpuNum : 1;// ��cpu�ĸ���Сһ��ʱ��ֻ��һ���߳�
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
							double tmp = 0;

							for (int i = 0; i < weights.length; i++) {
								tmp += weights[i][j] * lastOutputs[i];
							}

							outputs[j] = 1 / (1 + Math.pow(Math.E,
									-(tmp + theta[j])));

						}
						gate.countDown();
					}
				};
				runner.run(task);
			}

			gate.await();// �ȴ������߳�����һ������

		}
	}

	/**
	 * ��ȡ��index����񾭲㣬index��0��ʼ
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
			String line = in.readLine();// ��һ�б�ʾ�ж��ٲ�
			bp.layerNum = Integer.parseInt(line);
			bp.layerSize = Util.string2ints(in.readLine());
			bp.lables = Util.line2list(in.readLine());
			bp.maxLable = bp.lables.size() - 1;
			for (int i = 0; i < bp.layerNum; i++) {// ��ʼ��ÿһ��
				NeuralLayer layer = new NeuralLayer();
				layer.setSize(bp.layerSize[i]);// �����񾭵�Ԫ�ĸ���
				layer.setLayerIndex(i);// ���ò��
				line = in.readLine();
				if (!line.equals("null"))// ��ƫ��
					layer.setBiases(Util.string2doubles(line));
				line = in.readLine();
				if (!line.equals("null")) {// ��Ȩ��
					int[] weightSize = Util.string2ints(line);// Ȩ���ж����к���
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
	 * ����ģ��,ֻ��Ҫ����ƫ�ú�Ȩ�ؼ���
	 * 
	 * @param fileName
	 */
	public void saveModel(String fileName) {
		try {
			PrintWriter out = new PrintWriter(fileName);
			out.write(layerNum + "\n");// ��һ���ǲ���
			out.write(Util.array2string(layerSize) + "\n");
			out.write(Util.list2line(lables) + "\n");// �����������
			for (int i = 0; i < layerNum; i++) {// ����ÿһ��
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

	/**
	 * �������
	 * 
	 * @param fileName
	 */
	public void setLables(String fileName) {
		try {
			lables = new ArrayList<String>();
			BufferedReader reader = new BufferedReader(new FileReader(fileName));
			String line = reader.readLine();
			while (line != null) {
				lables.add(line.trim());
				line = reader.readLine();
			}

			maxLable = lables.size();
			System.out.println(lables);
			System.out.println(maxLable);
			reader.close();
		} catch (IOException e) {
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

}
