package bp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import data.Dataset.Record;

/**
 * 神经层
 * 
 * @author jiqunpeng
 * 
 *         创建时间：2014-6-13 下午3:26:55
 */
public class NeuralLayer {
	private static Random r = new Random(10);
	// 神经单元层号,第0层表示输入层
	private int layerIndex;
	// 神经元的个数
	private int size;
	// 上层神经元的个数
	private int lastSize;
	// 权重
	private double[][] weights;
	// 神经单元输出值
	private double[] outputs;
	// 各个神经元的偏置
	private double[] biases;
	// 各个神经单元的错误率
	private double[] errors;

	public NeuralLayer() {

	}

	public NeuralLayer(int layoutIndex, int size, int lastSize) {
		this.layerIndex = layoutIndex;
		this.size = size;
		this.lastSize = lastSize;
		outputs = new double[size];
		if (this.layerIndex > 0) {// 第一层输入层没有偏置和权重
			randomInitBias();
			randomInitWeight();
			errors = new double[size];// 2014年7月9日19:53:29
										// 修改输入层没有残差
		}
	}

	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("第");
		sb.append(layerIndex);
		sb.append("层，结点数目：");
		sb.append(size);
		sb.append("输出值：");
		sb.append(Arrays.toString(outputs));
		sb.append("\n");
		return sb.toString();
	}

	/**
	 * 该层神经单元的个数
	 * 
	 * @return
	 */
	public int getSize() {
		return size;
	}

	/**
	 * 设置神经单元的个数
	 * 
	 * @param size
	 */
	public void setSize(int size) {
		this.size = size;
	}

	/**
	 * 设置层号
	 * 
	 * @param index
	 */
	public void setLayerIndex(int index) {
		this.layerIndex = index;
	}

	/**
	 * 随机初始化权重值
	 */
	private void randomInitWeight() {
		weights = new double[lastSize][size];
		for (int i = 0; i < lastSize; i++) {
			for (int j = 0; j < size; j++) {
				weights[i][j] = r.nextDouble() - 0.5;
			}
		}
	}

	/***
	 * 随机初始化权重
	 */
	private void randomInitBias() {
		biases = new double[size];
		for (int i = 0; i < size; i++) {
			biases[i] = r.nextDouble();
		}
	}


	/**
	 * 设置输出
	 * 
	 * @param record
	 */
	public void setOutputs(Record record) {
		double[] attrs = record.getAttrs();
		for (int i = 0; i < attrs.length; i++) {
			outputs[i] = attrs[i];
		}
	}

	public double[] getErrors() {
		return errors;
	}

	public double[] getOutputs() {
		return outputs;
	}

	public double[][] getWeights() {
		return weights;
	}

	public double[] getBiases() {
		return biases;
	}

	public void setErrors(double[] errors) {
		this.errors = errors;
	}

	public void setOutputs(double[] outputs) {
		this.outputs = outputs;
	}

	public void setWeights(double[][] weights) {
		this.weights = weights;
	}

	public void setBiases(double[] biases) {
		this.biases = biases;
	}
}
