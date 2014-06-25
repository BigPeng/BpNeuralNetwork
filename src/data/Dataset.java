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
	// 保存数据
	private List<Record> records;
	// 类别列号下标
	private int lableColumnIndex;
	// 类别,根据类别id找对应的类别
	private List<String> lables;
	// 类别map，方便查找类别对应的id
	private Map<String, Integer> lablesMap;
	private Integer lablesId;

	public Dataset(int classIndex) {

		this.lableColumnIndex = classIndex;
		records = new ArrayList<Record>();
		if (lableColumnIndex != -1) {// 训练数据才有类别
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
	 * 清空数据
	 */
	public void clear() {
		records.clear();
	}

	/**
	 * 添加一个记录
	 * 
	 * @param attrs
	 *            记录的属性
	 * @param lable
	 *            记录的类标
	 */
	public void append(double[] attrs, String lable) {
		records.add(new Record(attrs, lable2id(lable)));
	}

	public Iterator<Record> iter() {
		return records.iterator();
	}

	/**
	 * 获取第index条记录的属性
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
	 * 导入数据集
	 * 
	 * @param filePath
	 *            文件名加路径
	 * @param tag
	 *            字段分隔符
	 * @param lableIndex
	 *            类标下标，从0开始
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
				if (lableIndex != -1) {// 训练数据，有类别
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
		System.out.println("导入数据:" + dataset.size());
		return dataset;
	}

	/**
	 * string类型的类别到id的转化
	 * 
	 * @param lable
	 * @return
	 */
	private Integer lable2id(String lable) {
		Integer lableId;
		if (lablesMap.containsKey(lable))// 前面加入的记录已经有了该类标
			lableId = lablesMap.get(lable);
		else {// 新类标
			lableId = lablesId++;
			lablesMap.put(lable, lableId);
			lables.add(lable);
		}
		return lableId;
	}

	/**
	 * 数据记录(实例),记录由属性和类别组成,类别必须为第一列或者最后一列或者空
	 * 
	 * @author jiqunpeng
	 * 
	 *         创建时间：2014-6-15 下午8:03:29
	 */
	public class Record {
		// 存储数据
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
		 * 该记录的属性
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
		 * 获取该记录的类标的下标
		 * 
		 * @return
		 */
		public Integer getLableId() {
			if (lableColumnIndex == -1)
				return null;
			return lableId;
		}

		/**
		 * 获取类别
		 * 
		 * @return
		 */
		public String getLable() {
			if (lableColumnIndex == -1)
				return null;
			return lables.get(lableId);
		}

		/**
		 * 对类标进行二进制编码
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
	 * 获取数据集的类别集合
	 * 
	 * @return
	 */
	public List<String> getLables() {
		return lables;
	}

}
