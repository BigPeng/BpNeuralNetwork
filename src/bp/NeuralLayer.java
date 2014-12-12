package bp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import data.Dataset.Record;

/**
 * �񾭲�
 * 
 * @author jiqunpeng
 * 
 *         ����ʱ�䣺2014-6-13 ����3:26:55
 */
public class NeuralLayer {
	private static Random r = new Random(10);
	// �񾭵�Ԫ���,��0���ʾ�����
	private int layerIndex;
	// ��Ԫ�ĸ���
	private int size;
	// �ϲ���Ԫ�ĸ���
	private int lastSize;
	// Ȩ��
	private double[][] weights;
	// �񾭵�Ԫ���ֵ
	private double[] outputs;
	// ������Ԫ��ƫ��
	private double[] biases;
	// �����񾭵�Ԫ�Ĵ�����
	private double[] errors;

	public NeuralLayer() {

	}

	public NeuralLayer(int layoutIndex, int size, int lastSize) {
		this.layerIndex = layoutIndex;
		this.size = size;
		this.lastSize = lastSize;
		outputs = new double[size];
		if (this.layerIndex > 0) {// ��һ�������û��ƫ�ú�Ȩ��
			randomInitBias();
			randomInitWeight();
			errors = new double[size];// 2014��7��9��19:53:29
										// �޸������û�вв�
		}
	}

	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("��");
		sb.append(layerIndex);
		sb.append("�㣬�����Ŀ��");
		sb.append(size);
		sb.append("���ֵ��");
		sb.append(Arrays.toString(outputs));
		sb.append("\n");
		return sb.toString();
	}

	/**
	 * �ò��񾭵�Ԫ�ĸ���
	 * 
	 * @return
	 */
	public int getSize() {
		return size;
	}

	/**
	 * �����񾭵�Ԫ�ĸ���
	 * 
	 * @param size
	 */
	public void setSize(int size) {
		this.size = size;
	}

	/**
	 * ���ò��
	 * 
	 * @param index
	 */
	public void setLayerIndex(int index) {
		this.layerIndex = index;
	}

	/**
	 * �����ʼ��Ȩ��ֵ
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
	 * �����ʼ��Ȩ��
	 */
	private void randomInitBias() {
		biases = new double[size];
		for (int i = 0; i < size; i++) {
			biases[i] = r.nextDouble();
		}
	}


	/**
	 * �������
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
