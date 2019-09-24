import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class Runner {
    public static void main(String[] args) throws Exception {
        new Runner().run();
    }

    private void run() throws Exception {
        int inputNodesNumber = 784;
        int hiddenNodesNumber = 70;
        int outputNodesNumber = 10;
        double learningRate = 0.3;
        NeuralNetwork n = new NeuralNetwork(inputNodesNumber, hiddenNodesNumber, outputNodesNumber, learningRate);

        List<String> trainingDataList = Files.readAllLines(Paths.get("C:\\Users\\fed\\Downloads\\mnist_train.csv"));
//        trainingDataList.stream().limit(1).forEach(System.out::println);

        trainingDataList.stream().limit(60000).forEach(record -> {
            String[] allValues = record.split(",");
            double[] inputs = Arrays.stream(allValues).skip(1).map(Integer::valueOf).mapToDouble(i -> (double) i / 255.0 * 0.99 + 0.01).toArray();
            double[] targets = new double[] { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
            targets[Integer.parseInt(allValues[0])] = 0.99;
            n.train(inputs, targets);
        });


        List<String> testingDataList = Files.readAllLines(Paths.get("C:\\Users\\fed\\Downloads\\mnist_test.csv"));
//        trainingDataList.stream().limit(1).forEach(System.out::println);

        List<Integer> scoreList = new ArrayList<>();
        testingDataList.stream().limit(10000).forEach(record -> {
            String[] allValues = record.split(",");
            int correctLabel = Integer.parseInt(allValues[0]);
//            System.out.println("Correct label: " + correctLabel);
            double[] inputs = Arrays.stream(allValues).skip(1).map(Integer::valueOf).mapToDouble(i -> (double) i / 255.0 * 0.99 + 0.01).toArray();
            double[] outputs = n.query(inputs);
//            Arrays.stream(outputs).forEach(it -> System.out.print(it + " "));
//            System.out.println();
            int label = findMaxElementIndex(outputs);
//            System.out.println("Correct label: " + correctLabel);
            if (label == correctLabel) {
                scoreList.add(1);
            } else {
                scoreList.add(0);
            }
        });
        System.out.println("Performance: " + (double) (scoreList.stream().mapToInt(x -> x).sum()) / scoreList.size());
    }

    private static int findMaxElementIndex(double[] array) {
        int maxElementIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxElementIndex]) {
                maxElementIndex = i;
            }
        }
        return maxElementIndex;
    }
}
