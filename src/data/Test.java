package data;

import java.util.Arrays;

public class Test {

	private double[] data;

	public Test() {
		data = new double[10];
	}
	
	public double[] getData(){
		return data;
	}
	
	public void print(){
		System.out.println(Arrays.toString(data));
	}
	
}
