import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class NeuralNetworkTest {

    @Test
    void transposeMatrix() {
        double[][] matrix = new double[][] {
                new double[] { 1, 2, 3, 4},
                new double[] { 5, 6, 7, 8}
        };
        double[][] result = NeuralNetwork.transposeMatrix(matrix);
        assertAll("",
                () -> assertArrayEquals(result[0], new double[]{ 1.0, 5.0 }),
                () -> assertArrayEquals(result[1], new double[]{ 2.0, 6.0 }),
                () -> assertArrayEquals(result[2], new double[]{ 3.0, 7.0 }),
                () -> assertArrayEquals(result[3], new double[]{ 4.0, 8.0 })
        );
    }

    @Test
    void multiplyMatrixes() {
        double[][] matrix1 = new double[][] {
                new double[] { 1, 2, 3, 4},
                new double[] { 5, 6, 7, 8}
        };
        double[][] matrix2 = new double[][] {
                new double[] {9, 10, 11},
                new double[] {12, 13, 14},
                new double[] {15, 16, 17},
                new double[] {18, 19, 20}
        };
        double[][] result = NeuralNetwork.multiplyMatrixes(matrix1, matrix2);
        assertAll("",
                () -> assertArrayEquals(result[0], new double[]{ 150.0, 160.0, 170.0}),
                () -> assertArrayEquals(result[1], new double[]{ 366.0, 392.0, 418.0})
        );
    }

    @Test
    void multiplyMatrixRowValues() {
        double[][] matrix1 = new double[][] {
                new double[] { 1, 2, 3, 4},
                new double[] { 5, 6, 7, 8}
        };
        double[][] matrix2 = new double[][]{
                new double[] {2, 0},
                new double[] {3, 0}
        };
        double[][] result = NeuralNetwork.multiplyMatrixRowValues(matrix1, matrix2);
        assertAll("",
                () -> assertArrayEquals(result[0], new double[]{ 2.0, 4.0, 6.0, 8.0}),
                () -> assertArrayEquals(result[1], new double[]{ 15.0, 18.0, 21.0, 24.0})
        );
    }

    @Test
    void updateMatrix() {
    }

    @Test
    void sumMatrixes() {
    }

    @Test
    void substractMatrixes() {
    }
}