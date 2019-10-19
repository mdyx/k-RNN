import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

public class KRNN {
	// data information
	private int n;
	private int m;
	private String trainRoad;
	private String resultRoad;
	
	// hyper parameter
	// predict N items for each user
	private int N;
	// k-nearest neighbor users
	private int K;
	// size of final neighborhood
	private int l;
	private double gamma;
	
	// model
	private Map<Integer, Set<Integer>> I_u;
	private Map<Integer, Set<Integer>> U_i;
	private Map<Integer, Map<Integer, Double>> Nl_u;
	private Set<Integer> I;
	
	// evaluation
	private String testRoad;
	private Map<Integer, Set<Integer>> ITe_u;
	
	public KRNN(String[] args) {
        // init
        n = 943;
        m = 1682;
        trainRoad = "";
        resultRoad = "";
        N = 5;
		K = 50;
		l = 50;
		gamma = 0.07;
        testRoad = "";
        
        // build
		I_u = new HashMap<Integer, Set<Integer>>();
		U_i = new HashMap<Integer, Set<Integer>>();
		Nl_u = new HashMap<Integer, Map<Integer, Double>>();
		I = new HashSet<Integer>();
		ITe_u = new HashMap<Integer, Set<Integer>>();
        
		// init
		for (int k = 0; k < args.length; k++)
        {
			switch (args[k]) {
			// data information
    		case "-n":
				n = Integer.parseInt(args[++k]);
				break;
    		case "-m":
				m = Integer.parseInt(args[++k]);
				break;
    		case "-trainRoad":
				trainRoad = args[++k];
				break;
    		case "-resultRoad":
				resultRoad = args[++k];
				break;
			
			// hyper parameter
    		case "-N":
				N = Integer.parseInt(args[++k]);
				break;
    		case "-K":
				K = Integer.parseInt(args[++k]);
				break;
    		case "-l":
				l = Integer.parseInt(args[++k]);
				break;
			case "-gamma":
				gamma = Double.parseDouble(args[++k]);
				break;
			
			// evaluation
			case "-testRoad":
				testRoad = args[++k];
				break;
            
            default:
                System.out.println("args " + args[k] + " error!");
			}
        }
	}
	
	private void readData() throws IOException {
		// read train data
		BufferedReader bufferedReader = new
		BufferedReader(new FileReader(trainRoad));
		String line = null;
		
		while ((line = bufferedReader.readLine()) != null) {
    		String[] terms = line.split("\\s+|,|;");
    		int user = Integer.parseInt(terms[0]);
    		int item = Integer.parseInt(terms[1]);
			
			if (!I_u.containsKey(user)) {
				I_u.put(user, new HashSet<Integer>());
			}
			I_u.get(user).add(item);
			if (!U_i.containsKey(item)) {
				U_i.put(item, new HashSet<Integer>());
			}
			U_i.get(item).add(user);
			
			I.add(item);
		}
		
		bufferedReader.close();
		
		// read test data
		bufferedReader = new BufferedReader(new FileReader(testRoad));
		while ((line = bufferedReader.readLine()) != null) {
    		String[] terms = line.split("\\s+|,|;");
    		int user = Integer.parseInt(terms[0]);
    		int item = Integer.parseInt(terms[1]);
			
			if (!ITe_u.containsKey(user)) {
				ITe_u.put(user, new HashSet<Integer>());
			}
			ITe_u.get(user).add(item);
			
			I.add(item);
		}
		
		bufferedReader.close();
	}
	
	private void train() {
		// bulid Nk_u
		Map<Integer, Set<Integer>> Nk_u = new HashMap<Integer, Set<Integer>>();
		
		for (int u: I_u.keySet()) {
			Map<Integer, Double> userToSim = new HashMap<Integer, Double>();
			
			for (int w: I_u.keySet()) {
				if (u == w) {
					continue;
				}
				
				int sum = 0;
				Set<Integer> itemOfU = I_u.get(u);
				Set<Integer> itemOfW = I_u.get(w);
				for (int i: itemOfU) {
					if (itemOfW.contains(i)) {
						sum++;
					}
				}
				if (sum == 0) {
					continue;
				}
				
				double sim = (double)sum / (itemOfU.size() + itemOfW.size() - sum);
				userToSim.put(w, sim);
			}
			
			List<Map.Entry<Integer, Double>> simList = new ArrayList<Map.Entry<Integer, Double>>(userToSim.entrySet());
			Collections.sort(simList, new Comparator<Map.Entry<Integer, Double>>() {
				public int compare(Entry<Integer, Double> o1, Entry<Integer, Double> o2) {
					return -o1.getValue().compareTo(o2.getValue());
				}
			});
			
			// K-reciprocal
			if (simList.size() > K) {
				simList = simList.subList(0, K);
			}
			Set<Integer> neighborUsers = new HashSet<Integer>();
			for (Map.Entry<Integer, Double> entry: simList) {
				int w = entry.getKey();
				neighborUsers.add(w);
			}
			Nk_u.put(u, neighborUsers);
		}
		
		// bulid Nl_u
		for (int u: I_u.keySet()) {
			Map<Integer, Double> userToSim = new HashMap<Integer, Double>();
				
			for (int w: I_u.keySet()) {
				if (u == w) {
					continue;
				}
					
				int sum = 0;
				Set<Integer> itemOfU = I_u.get(u);
				Set<Integer> itemOfW = I_u.get(w);
				for (int i: itemOfU) {
					if (itemOfW.contains(i)) {
						sum++;
					}
				}
				if (sum == 0) {
					continue;
				}
				
				double sim = (double)sum / (itemOfU.size() + itemOfW.size() - sum);
				if (Nk_u.containsKey(u) && Nk_u.get(u).contains(w) && Nk_u.containsKey(w) && Nk_u.get(w).contains(u)) {
					sim *= (1 + gamma);
				}
				userToSim.put(w, sim);
			}
			
			List<Map.Entry<Integer, Double>> simList = new ArrayList<Map.Entry<Integer, Double>>(userToSim.entrySet());
			Collections.sort(simList, new Comparator<Map.Entry<Integer, Double>>() {
				public int compare(Entry<Integer, Double> o1, Entry<Integer, Double> o2) {
					return -o1.getValue().compareTo(o2.getValue());
				}
			});
			
			if (simList.size() > l) {
				simList = simList.subList(0, l);
			}
			Map<Integer, Double> knnOfU = new HashMap<Integer, Double>();
			for (Map.Entry<Integer, Double> entry2: simList) {
				int w = entry2.getKey();
				double sim = entry2.getValue();
				knnOfU.put(w, sim);
			}
			Nl_u.put(u, knnOfU);
		}
	}
	
    private void outputInformation(PrintWriter printWriter) {
        printWriter.println("n = " + n);
        printWriter.println("m = " + m);
        printWriter.println("N = " + N);
        printWriter.println("K = " + K);
        printWriter.println("l = " + l);
        printWriter.println("gamma = " + gamma);
        printWriter.println("trainRoad = " + trainRoad);
		printWriter.println("testRoad = " + testRoad);
        printWriter.println("resultRoad = " + resultRoad);
        printWriter.println("");
    }
    
	private void predictAndEvaluation(PrintWriter printWriter, boolean isResult)
    throws IOException {
		double[] precisionSum = new double[N + 1];
		double[] recallSum = new double[N + 1];
		double[] F1Sum = new double[N + 1];
		double[] NDCGSum = new double[N + 1];
		double[] oneCallSum = new double[N + 1];
		double MRRSum = 0;
		double MAPSum = 0;
		double ARPSum = 0;
		double AUCSum = 0;
		
		double[] DCGbest = new double[N + 1];
		DCGbest[0] = 0;
		for (int k = 1; k <= N; k++) {
			DCGbest[k] = DCGbest[k - 1] + 1 / Math.log(k + 1);
		}
        
		// u may be missing, but need to predict the missing u
		// i may be missing, and need to skip the missing i when predict
		int testUserNum = 0;
    	for(int u = 1; u <= n; u++) {
			// don't need to evaluation if test data not contains user u
    		if (!I_u.containsKey(u) || !ITe_u.containsKey(u)) {
    			continue;
			}
			testUserNum++;
    		
    		Set<Integer> trainItemSetOfU = new HashSet<Integer>();
    		if (I_u.containsKey(u)) {
    			trainItemSetOfU = I_u.get(u);
    		}
			
    		Set<Integer> testItemSetOfU = ITe_u.get(u);
    		int testItemNumOfU = testItemSetOfU.size();
    		
    		// prediction rating
			List<Map.Entry<Integer, Double>> simList = new ArrayList<Map.Entry<Integer, Double>>();
			if (Nl_u.containsKey(u)) {
				simList = new ArrayList<Map.Entry<Integer, Double>>(Nl_u.get(u).entrySet());
		        Collections.sort(simList, new Comparator<Map.Entry<Integer, Double>>() {
		            public int compare(Entry<Integer, Double> o1, Entry<Integer, Double> o2) {
		                return -o1.getValue().compareTo(o2.getValue());
		            }
		        });
			}
			
    		HashMap<Integer, Double> itemToHatR = new HashMap<Integer, Double>();
    		for (int i: I) {
				if (trainItemSetOfU.contains(i)) {
					continue;
				}
				itemToHatR.put(i, 0.0);
			}
			
			for (Map.Entry<Integer, Double> entry: simList) {
				int w = entry.getKey();
				double sim = entry.getValue();
				
				for (int item: I_u.get(w)) {
					if (I_u.get(u).contains(item)) {
						continue;
					}
					
					itemToHatR.put(item, itemToHatR.get(item) + sim);
				}
			}
			
			// sort
    		List<Map.Entry<Integer,Double>> list =
			new ArrayList<Map.Entry<Integer,Double>>(itemToHatR.entrySet());
    		Collections.sort(list, new Comparator<Map.Entry<Integer,Double>>() {
    			public int compare(Map.Entry<Integer, Double> o1,
                Map.Entry<Integer, Double> o2) {
    				return o2.getValue().compareTo(o1.getValue());
    			}
    		});
    		
			// top-k list
    		Iterator<Map.Entry<Integer, Double>> mapIter = list.iterator();
    		int k = 1;
            List<Integer> topKResult = new ArrayList<Integer>();
    		while (mapIter.hasNext() && k <= N) {
    			Map.Entry<Integer, Double> entry = mapIter.next(); 
    			int item = entry.getKey();
                topKResult.add(item);
    		}
			
    		// evaluation
    		// precision, recall, F1, NDCG, 1-call
    		int hitNum = 0;
    		double[] DCG = new double[N + 1];
    		double[] DCGbest2 = new double[N + 1];
    		for(k = 1; k <= N; k++) {
    			DCG[k] = DCG[k - 1];
    			int item = topKResult.get(k - 1);
    			if (testItemSetOfU.contains(item)) {
        			hitNum += 1;
        			DCG[k] += 1 / Math.log(k + 1);
    			}
				
    			double prec = (double)hitNum / k;
    			double rec = (double)hitNum / testItemNumOfU;    			
    			double F1 = 0;
    			if (prec + rec > 0) {
    				F1 = 2 * prec * rec / (prec + rec);
				}
    			precisionSum[k] += prec;
    			recallSum[k] += rec;
    			F1Sum[k] += F1;
				
    			if (testItemNumOfU >= k) {
    				DCGbest2[k] = DCGbest[k];
				}
    			else {
    				DCGbest2[k] = DCGbest2[k - 1];
				}
    			NDCGSum[k] += DCG[k] / DCGbest2[k];
    			oneCallSum[k] += hitNum > 0 ? 1 : 0; 
    		}
    		
    		// MRR
			int p = 1;
			mapIter = list.iterator();
			while (mapIter.hasNext())
			{
				Map.Entry<Integer, Double> entry = mapIter.next(); 
				int item = entry.getKey();
				
				if(testItemSetOfU.contains(item)) {
					break;
				}
				p += 1;
			}
			MRRSum += 1.0 / p;
    		
			// MAP
			p = 1;
			double AP = 0;
			int hitBefore = 0;
			mapIter = list.iterator();
			while (mapIter.hasNext())
			{	
				Map.Entry<Integer, Double> entry = mapIter.next(); 
				int item = entry.getKey();
				
				if(testItemSetOfU.contains(item)) {
					hitBefore += 1;
					AP += hitBefore / (double)p;
				}
				p += 1;
			}
			MAPSum += AP / testItemNumOfU;
    		
			// ARP
			p = 1;
			double RP = 0;    		
			mapIter = list.iterator();    		
			while (mapIter.hasNext()) {
				Map.Entry<Integer, Double> entry = mapIter.next(); 
				int item = entry.getKey();
				
				if(testItemSetOfU.contains(item)) {
					RP += p;
				}
				p += 1;
			}
			ARPSum += RP / itemToHatR.size() / testItemNumOfU;
			
			// AUC
			Map<Integer,Integer> itemToCnt = new HashMap<Integer,Integer>();
			for (int item: testItemSetOfU) {
				itemToCnt.put(item, 0);
			}
			
			for(int j: itemToHatR.keySet()) {
				double r_uj = itemToHatR.get(j);
				if(!testItemSetOfU.contains(j)) {
					for (int i: testItemSetOfU) {
						double r_ui = itemToHatR.get(i);
						if (r_ui > r_uj) {
							int c = itemToCnt.get(i) + 1;
							itemToCnt.put(i, c);
						}
					}
				}
			}
			
			int AUC = 0;
			for (int i: testItemSetOfU) {
				AUC += itemToCnt.get(i);
			}
			AUCSum += (double) AUC / (itemToHatR.size() - testItemNumOfU) / testItemNumOfU;
    	}
        
		// output evaluation result
    	for(int k = 1; k <= N; k++) {
    		double prec = precisionSum[k] / testUserNum;
    		printWriter.println("Prec@" + Integer.toString(k) + ":" + Double.toString(prec));
    	}
    	for(int k = 1; k <= N; k++) {
    		double rec = recallSum[k] / testUserNum;
    		printWriter.println("Rec@" + Integer.toString(k) + ":" + Double.toString(rec));
    	}
    	for(int k = 1; k <= N; k++) {
    		double F1 = F1Sum[k] / testUserNum;
    		printWriter.println("F1@" + Integer.toString(k) + ":" + Double.toString(F1));
    	}
    	for(int k = 1; k <= N; k++) {
    		double NDCG = NDCGSum[k] / testUserNum;
    		printWriter.println("NDCG@" + Integer.toString(k)+":"+Double.toString(NDCG));
    	}
    	for(int k = 1; k <= N; k++) {
    		double oneCall = oneCallSum[k] / testUserNum;
    		printWriter.println("1-call@" + Integer.toString(k) + ":" + Double.toString(oneCall));
    	}
    	double MRR = MRRSum / testUserNum;
    	printWriter.println("MRR:" + Double.toString(MRR));
    	double MAP = MAPSum / testUserNum;
    	printWriter.println("MAP:" + Double.toString(MAP));
    	double ARP = ARPSum / testUserNum;
    	printWriter.println("ARP:" + Double.toString(ARP));
    	double AUC = AUCSum / testUserNum;
    	printWriter.println("AUC:" + Double.toString(AUC));
        printWriter.println("");
	}
    
    private void outputResult(long startTime) throws IOException {
        PrintWriter printWriter = new PrintWriter(resultRoad, "UTF-8");
        outputInformation(printWriter);
        predictAndEvaluation(printWriter, true);
		printWriter.println("use " + (double)(System.currentTimeMillis() - startTime) / 1000 / 60 / 60 + " h(s)");
		printWriter.println("");
        printWriter.close();
    }
    
	public static void main(String[] args) {
		long startTime = System.currentTimeMillis();
		try {
			KRNN krnn = new KRNN(args);
			krnn.readData();
			krnn.train();
            krnn.outputResult(startTime);
		}
		catch (IOException ioException) {
			System.out.println("io error!");
		}
		System.out.println("use " + (double)(System.currentTimeMillis() - startTime) / 1000 / 60 / 60 + " h(s)");
	}
}
