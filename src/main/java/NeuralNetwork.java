import java.util.Random;
import java.util.function.DoubleUnaryOperator;

public class NeuralNetwork {
    private int inputNodesNumber;
    private int hiddenNodesNumber;
    private int outputNodesNumber;
    private double learningRate;
    private double[][] wih;
    private double[][] who;
    private DoubleUnaryOperator activationFunction = NeuralNetwork::sigmoid;

    public NeuralNetwork(int inputNodesNumber, int hiddenNodesNumber, int outputNodesNumber, double learningRate) {
        this.inputNodesNumber = inputNodesNumber;
        this.hiddenNodesNumber = hiddenNodesNumber;
        this.outputNodesNumber = outputNodesNumber;
        this.learningRate = learningRate;

        this.wih = new double[hiddenNodesNumber][inputNodesNumber];
        Random randomNumbersGenerator1 = new Random();
        for (int i = 0; i < hiddenNodesNumber; i++) {
            for (int l = 0; l < inputNodesNumber; l++) {
                wih[i][l] = randomNumbersGenerator1.nextGaussian();
            }
        }

        this.who = new double[outputNodesNumber][hiddenNodesNumber];
        Random randomNumbersGenerator2 = new Random();
        for (int i = 0; i < outputNodesNumber; i++) {
            for (int l = 0; l < hiddenNodesNumber; l++) {
                who[i][l] = randomNumbersGenerator2.nextGaussian();
            }
        }
    }

    public void train(double[] inputsData, double[] targetsData) {
        double[][] inputs = transposeMatrix(new double[][] { inputsData });
        double[][] targets = transposeMatrix(new double[][] { targetsData });
        double[][] hiddenInputs = multiplyMatrixes(wih, inputs);
        double[][] hiddenOutputs = updateMatrix(hiddenInputs, activationFunction);
        double[][] finalInputs = multiplyMatrixes(who, hiddenOutputs);
        double[][] finalOutputs = updateMatrix(finalInputs, activationFunction);

        double[][] outputErrors = substractMatrixes(targets, finalOutputs);
        double[][] hiddenErrors = multiplyMatrixes(transposeMatrix(who), outputErrors);

        who = sumMatrixes(who, multiplyMatrixes(multiplyMatrixRowValues(outputErrors, multiplyMatrixRowValues(finalOutputs, updateMatrix(updateMatrix(finalOutputs, x -> -x), x -> x + 1))), transposeMatrix(hiddenOutputs)));
        wih = sumMatrixes(wih, multiplyMatrixes(multiplyMatrixRowValues(hiddenErrors, multiplyMatrixRowValues(hiddenOutputs, updateMatrix(updateMatrix(hiddenOutputs, x -> -x), x -> x + 1))), transposeMatrix(inputs)));
    }

    public double[] query(double[] inputsData) {
        double[][] inputs = transposeMatrix(new double[][] { inputsData });
        double[][] hiddenInputs = multiplyMatrixes(wih, inputs);
        double[][] hiddenOutputs = updateMatrix(hiddenInputs, activationFunction);
        double[][] finalInputs = multiplyMatrixes(who, hiddenOutputs);
        double[][] finalOutputs = updateMatrix(finalInputs, activationFunction);

        return transposeMatrix(finalOutputs)[0];
    }

    public static double sigmoid(double x) {
        return (1/( 1 + Math.pow(Math.E,(-1*x))));
    }

    public static double[][] transposeMatrix(double[][] matrix) {
        int height = matrix.length;
        int width = matrix[0].length;
        double[][] transposedMatrix = new double[width][height];

        for (int i = 0; i < width; i++) {
            for (int l = 0; l < height; l++) {
                transposedMatrix[i][l] = matrix[l][i];
            }
        }

        return transposedMatrix;
    }

    public static double[][] multiplyMatrixes(double[][] matrix1, double[][] matrix2) {
        int height1 = matrix1.length;
        int height2 = matrix2.length;
        int width1 = matrix1[0].length;
        int width2 = matrix2[0].length;

        double[][] resultMatrix = new double[height1][width2];
        for (int i = 0; i < height1; i++) {
            for (int l = 0; l < width2; l++) {
                resultMatrix[i][l] = 0.0;
                for (int j = 0; j < height2; j++) {
                    resultMatrix[i][l] += matrix1[i][j] * matrix2[j][l];
                }
            }
        }

        return resultMatrix;
    }

    public static double[][] multiplyMatrixRowValues(double[][] matrix1, double[][] matrix2) {
        int height1 = matrix1.length;
        int height2 = matrix2.length;
        int width1 = matrix1[0].length;
        int width2 = matrix2[0].length;

        double[][] resultMatrix = new double[height1][width1];
        for (int i = 0; i < height1; i++) {
            for (int l = 0; l < width1; l++) {
                resultMatrix[i][l] = matrix1[i][l] * matrix2[i][0];
            }
        }

        return resultMatrix;
    }

    public static double[][] updateMatrix(double[][] matrix, DoubleUnaryOperator updateFunction) {
        int height = matrix.length;
        int width = matrix[0].length;

        double[][] newMatrix = new double[height][width];
        for (int i = 0; i < height; i++) {
            for (int l = 0; l < width; l++) {
                newMatrix[i][l] = updateFunction.applyAsDouble(matrix[i][l]);
            }
        }

        return newMatrix;
    }

    public static double[][] sumMatrixes(double[][] matrix1, double[][] matrix2) {
        int height = matrix1.length;
        int width = matrix1[0].length;

        double[][] newMatrix = new double[height][width];
        for (int i = 0; i < height; i++) {
            for (int l = 0; l < width; l++) {
                newMatrix[i][l] = matrix1[i][l] + matrix2[i][l];
            }
        }

        return newMatrix;
    }

    public static double[][] substractMatrixes(double[][] matrix1, double[][] matrix2) {
        int height = matrix1.length;
        int width = matrix1[0].length;

        double[][] newMatrix = new double[height][width];
        for (int i = 0; i < height; i++) {
            for (int l = 0; l < width; l++) {
                newMatrix[i][l] = matrix1[i][l] - matrix2[i][l];
            }
        }

        return newMatrix;
    }
}
