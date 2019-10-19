Introduction:
File KRNN.java is the code of kRNN.

An example of run script:
javac KRNN.java
java KRNN -n 943 -m 1682 -trainRoad ML100K-copy1-train -resultRoad result -N 5 -K 100 -l 110 -gamma 0.05 -testRoad ML100K-copy1-test

where:
trainRoad: the road of training data. Each line of training data is required to follow the format "userID itemID".
resultRoad: the road of result.
testRoad:  the road of test data. Each line of training data is required to follow the format "userID itemID".
n, m, N, K, l, gamma: meanings of these parameter are in the paper.
