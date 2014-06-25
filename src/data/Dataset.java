package data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class Dataset {
	// ��������
	private List<Record> records;
	// ����к��±�
	private int lableColumnIndex;
	// ���,�������id�Ҷ�Ӧ�����
	private List<String> lables;
	// ���map�������������Ӧ��id
	private Map<String, Integer> lablesMap;
	private Integer lablesId;

	public Dataset(int classIndex) {

		this.lableColumnIndex = classIndex;
		records = new ArrayList<Record>();
		if (lableColumnIndex != -1) {// ѵ�����ݲ������
			lables = new ArrayList<String>();
			lablesMap = new HashMap<String, Integer>();
			lablesId = 0;
		}
	}

	public int size() {
		return records.size();
	}

	public int getLableColumnIndex() {
		return lableColumnIndex;
	}

	public void append(Record record) {
		records.add(record);
	}

	/**
	 * �������
	 */
	public void clear() {
		records.clear();
	}

	/**
	 * ���һ����¼
	 * 
	 * @param attrs
	 *            ��¼������
	 * @param lable
	 *            ��¼�����
	 */
	public void append(double[] attrs, String lable) {
		records.add(new Record(attrs, lable2id(lable)));
	}

	public Iterator<Record> iter() {
		return records.iterator();
	}

	/**
	 * ��ȡ��index����¼������
	 * 
	 * @param index
	 * @return
	 */
	public double[] getAttrs(int index) {
		return records.get(index).getAttrs();

	}

	public Integer getLable(int index) {
		return records.get(index).getLableId();
	}

	/**
	 * �������ݼ�
	 * 
	 * @param filePath
	 *            �ļ�����·��
	 * @param tag
	 *            �ֶηָ���
	 * @param lableIndex
	 *            ����±꣬��0��ʼ
	 * @return
	 */
	public static Dataset load(String filePath, String tag, int lableIndex) {
		Dataset dataset = new Dataset(lableIndex);
		File file = new File(filePath);
		try {

			BufferedReader in = new BufferedReader(new FileReader(file));
			String line;
			while ((line = in.readLine()) != null) {
				String[] datas = line.split(tag);
				if (datas.length == 0)
					continue;
				double[] data;
				if (lableIndex != -1)
					data = new double[datas.length - 1];
				else
					data = new double[datas.length];
				int i;
				for (i = 0; i < datas.length - 1; i++)
					data[i] = Double.parseDouble(datas[i]);
				Record record;
				if (lableIndex != -1) {// ѵ�����ݣ������
					Integer lableId = dataset.lable2id(datas[i]);
					record = dataset.new Record(data, lableId);
				} else {
					data[i] = Double.parseDouble(datas[i]);
					record = dataset.new Record(data);
				}
				dataset.append(record);
			}
			in.close();

		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
		System.out.println("��������:" + dataset.size());
		return dataset;
	}

	/**
	 * string���͵����id��ת��
	 * 
	 * @param lable
	 * @return
	 */
	private Integer lable2id(String lable) {
		Integer lableId;
		if (lablesMap.containsKey(lable))// ǰ�����ļ�¼�Ѿ����˸����
			lableId = lablesMap.get(lable);
		else {// �����
			lableId = lablesId++;
			lablesMap.put(lable, lableId);
			lables.add(lable);
		}
		return lableId;
	}

	/**
	 * ���ݼ�¼(ʵ��),��¼�����Ժ�������,������Ϊ��һ�л������һ�л��߿�
	 * 
	 * @author jiqunpeng
	 * 
	 *         ����ʱ�䣺2014-6-15 ����8:03:29
	 */
	public class Record {
		// �洢����
		private double[] attrs;
		private Integer lableId;

		private Record(double[] attrs, Integer lableId) {
			this.attrs = attrs;
			this.lableId = lableId;
		}

		public Record(double[] data) {
			attrs = data;
		}

		/**
		 * �ü�¼������
		 * 
		 * @return
		 */
		public double[] getAttrs() {
			return attrs;
		}

		public String toString() {
			StringBuilder sb = new StringBuilder();
			sb.append("attrs:");
			sb.append(Arrays.toString(attrs));
			sb.append("lable:");
			sb.append(lableId);
			return sb.toString();
		}

		/**
		 * ��ȡ�ü�¼�������±�
		 * 
		 * @return
		 */
		public Integer getLableId() {
			if (lableColumnIndex == -1)
				return null;
			return lableId;
		}

		/**
		 * ��ȡ���
		 * 
		 * @return
		 */
		public String getLable() {
			if (lableColumnIndex == -1)
				return null;
			return lables.get(lableId);
		}

		/**
		 * �������ж����Ʊ���
		 * 
		 * @param n
		 * @return
		 */
		public int[] getEncodeTarget(int n) {
			String binary = Integer.toBinaryString(lableId.intValue());
			byte[] bytes = binary.getBytes();
			int[] encode = new int[n];
			int j = n;
			for (int i = bytes.length - 1; i >= 0; i--)
				encode[--j] = bytes[i] - '0';

			return encode;
		}

		public double[] getDoubleEncodeTarget(int n) {
			String binary = Integer.toBinaryString(lableId.intValue());
			byte[] bytes = binary.getBytes();
			double[] encode = new double[n];
			int j = n;
			for (int i = bytes.length - 1; i >= 0; i--)
				encode[--j] = bytes[i] - '0';

			return encode;
		}

	}

	public static void main(String[] args) {
		Dataset d = new Dataset(10);

		Record r = d.new Record(new double[] { 3, 2, 2, 5, 4, 5, 3, 11, 3, 12,
				1 });
		int[] encode = r.getEncodeTarget(4);

		System.out.println(r.lableId);
		System.out.println(Arrays.toString(encode));
	}

	/**
	 * ��ȡ���ݼ�����𼯺�
	 * 
	 * @return
	 */
	public List<String> getLables() {
		return lables;
	}

}
