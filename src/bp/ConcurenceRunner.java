package bp;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * �������й���
 * 
 * @author jiqunpeng
 * 
 *         ����ʱ�䣺2014-6-16 ����3:33:41
 */
public class ConcurenceRunner {

	private static ExecutorService exec;
	public static int cpuNum;
	static {
		cpuNum = Runtime.getRuntime().availableProcessors();
		exec = Executors.newFixedThreadPool(cpuNum);
	}

	public void run(Runnable task) {
		exec.execute(task);
	}

	public static void stop() {
		exec.shutdown();
	}

}
