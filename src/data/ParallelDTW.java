package data;



import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public class ParallelDTW {
	private static final String TRAIN_PATH = "data/3510_make.txt";
	private static final String TEST_PATH = "data/xiong.txt";
	private static final int COLUMN_LENGTH = 1025;
	private static final int WINDOW_WIDTH = (int) (COLUMN_LENGTH * 0.005);
	
	public static void main(String[] args) {
		/*
		 * Read data of train set and data of test set 
		 * then store them in ArrayLists respectively
		 */
		List<Double[]> trainSet = readDataToArray(TRAIN_PATH);
		List<Double[]> testSet = readDataToArray(TEST_PATH);
		List<String> words = Util.readAsList("data/3500.words");
		
		// Record the DTW distance between one test set instance 
		// and every train set instance
		double[] dtwDistance = new double[trainSet.size()];
		double accuracyRate = 0.0;
		long startTime = System.currentTimeMillis();
		for (int i = 0; i < testSet.size(); ++i) {
			dtw(trainSet, testSet.get(i), dtwDistance);
			// classify the test data set using KNN method
			int minDtwDistIndex = 0;
			for (int j = 1; j < dtwDistance.length; j++) {
				if (dtwDistance[j] < dtwDistance[minDtwDistIndex])
					minDtwDistIndex = j;
			}
			Log.i(i+"");
			int originTab = testSet.get(i)[COLUMN_LENGTH-1].intValue();
			int classTab = trainSet.get(minDtwDistIndex)[COLUMN_LENGTH-1].intValue();
			
			 System.out.println(Arrays.toString(testSet.get(i)));
			 System.out.println(Arrays.toString(trainSet.get(minDtwDistIndex)));
			 System.out.print("class = " + classTab+":"+words.get(classTab));
			 System.out.print("; origin = " + originTab);
		
			 System.out.println("; " + (originTab == classTab));

			if (originTab == classTab) {
				accuracyRate += 1;
			}
		}
		long endTime = System.currentTimeMillis();
		
		System.out.println(accuracyRate / testSet.size());
		System.out.println("Time taken: " + (endTime - startTime));
		
	}
	
	/**
	 * Read the raw data from file and store them in an ArrayList
	 * @param path: file path
	 * @return an ArrayList
	 */
	public static List<Double[]> readDataToArray(String path) {
		List<Double[]> result = new ArrayList<>();
		
		try (BufferedReader br = new BufferedReader(new FileReader(path));) 
		{
			while (true) {
				String[] stringLine = br.readLine().split(",");
				Double[] array = new Double[COLUMN_LENGTH];
				int i = 0, j = 0;
				while (i < stringLine.length && j < COLUMN_LENGTH) {
					if (stringLine[i].length() > 0) {
						array[j++] = Double.valueOf(stringLine[i]);
					}
					++i;
				}
				result.add(array);
			}
		} catch (NullPointerException e) {
			System.out.println("Reading from " + path + " done.");
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return result;
	}
	
	public static void dtw(List<Double[]> trainSet, Double[] oneTest, double[] dtwDistance) {
		RecursiveAction mainTask = new DTWTask(trainSet, oneTest, dtwDistance, 0, trainSet.size());
		ForkJoinPool pool = new ForkJoinPool();
		pool.invoke(mainTask);
	}
	
	private static class DTWTask extends RecursiveAction {
		private static final long serialVersionUID = 1L;
		private final static int THRESHOLD = 875;
		private List<Double[]> trainSet;
		private Double[] oneTest;
		private double[] dtwDistance;
		private int start;
		private int end;
		
		public DTWTask(List<Double[]> trainSet, Double[] oneTest, 
				double[] dtwDistance, int start, int end) {
			this.trainSet = trainSet;
			this.oneTest = oneTest;
			this.dtwDistance = dtwDistance;
			this.start = start;
			this.end = end;
		}
		
		@Override
		protected void compute() {
			if (end - start <= THRESHOLD) {
				double[][] dtwMetrix = new double[COLUMN_LENGTH + 1][COLUMN_LENGTH + 1];
				
				// Assign the Double.MAX_VALUE to every element in the matrix
				// and assign 0 to the top-left corner element.
				// This help to calculate DTW
				for (int i = 0; i < dtwMetrix.length; i++) {
					for (int j = 0; j < dtwMetrix[0].length; j++) {
						dtwMetrix[i][j] = Double.MAX_VALUE;
					}
				}				
				dtwMetrix[0][0] = 0;
				
				for (int i = start; i < end; i++) {
					// calculate the distance between any two points
					for (int j = 1; j < WINDOW_WIDTH + 1; j++) {
						for (int k = 1; k < j + WINDOW_WIDTH  + 1; k++) {
							dtwMetrix[j][k] = Math.pow(oneTest[j - 1] - trainSet.get(i)[k -1], 2);
							// Calculate the DTW
							dtwMetrix[j][k] += minInThree(dtwMetrix[j-1][k-1], 
									  					  dtwMetrix[j-1][k], 
									  					  dtwMetrix[j][k-1]);
						}
					}

					for (int j = WINDOW_WIDTH + 1; j < dtwMetrix.length - WINDOW_WIDTH; j++) {
						for (int k = j - WINDOW_WIDTH; k < j + WINDOW_WIDTH + 1; k++) {
							dtwMetrix[j][k] = Math.pow(oneTest[j - 1] - trainSet.get(i)[k -1], 2);
							dtwMetrix[j][k] += minInThree(dtwMetrix[j-1][k-1], 
									  					  dtwMetrix[j-1][k], 
									  					  dtwMetrix[j][k-1]);
						}
					}
					
					for (int j = dtwMetrix.length - WINDOW_WIDTH; j < dtwMetrix.length; j++) {
						for (int k = j - WINDOW_WIDTH; k < dtwMetrix.length; k++) {
							dtwMetrix[j][k] = Math.pow(oneTest[j - 1] - trainSet.get(i)[k -1], 2);
							dtwMetrix[j][k] += minInThree(dtwMetrix[j-1][k-1], 
									  					  dtwMetrix[j-1][k], 
									  					  dtwMetrix[j][k-1]);
						}
					}
					
					int cnt = 0;
					for (int j = dtwMetrix.length - 1, k = dtwMetrix[0].length -1; j > 0 && k > 0;) {
						cnt++;
						if (dtwMetrix[j-1][k-1] <= Math.min(dtwMetrix[j-1][k], dtwMetrix[j][k-1])) {
							j--;
							k--;
						} else if (dtwMetrix[j-1][k] <= dtwMetrix[j][k-1]) {
							j--;
						} else {
							k--;
						}
					}
					dtwDistance[i] = Math.sqrt(dtwMetrix[COLUMN_LENGTH][COLUMN_LENGTH]) / cnt;
				}
			} else {
				int mid = (start + end) / 2;
				DTWTask first = new DTWTask(trainSet, oneTest, dtwDistance, start, mid);
				DTWTask second = new DTWTask(trainSet, oneTest, dtwDistance, mid, end);
				invokeAll(first, second);
			}
		}
		
	}
	
	public static double minInThree(double a, double b, double c) {
		return Math.min(a, Math.min(b, c));
	}

}