package bp;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * 并发运行工具
 * 
 * @author jiqunpeng
 * 
 *         创建时间：2014-6-16 下午3:33:41
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
