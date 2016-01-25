/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekateste;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.misc.HyperPipes;
import weka.core.Instances;

/**
 *
 * @author charles
 */
public class Wekateste {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        StringBuilder sb = new StringBuilder();
        Classifier classifier = null;
        SMO smo = new SMO();
        String dataset = "";

        HyperPipes hy = new HyperPipes();

        IBk ibk = new IBk(5);

        if (args.length != 0) {
            if ("smo".equals(args[0])) {
                classifier = smo;
            }
            if ("hy".equals(args[0])) {
                classifier = hy;
            }
            if ("ibk".equals(args[0])) {
                classifier = ibk;
            }
            dataset = args[1];

        } else {
            classifier = hy;
            dataset = "resultadoPretext.arff";

        }
        System.out.println("Classificador: " + classifier);
        System.out.println("Dataset: " + dataset);
//        classifier.buildClassifier(trainset);
        BufferedReader datafile = readDataFile(dataset);

        Instances data;
        Evaluation eval;
        try {
            data = new Instances(datafile);
            data.setClassIndex(data.numAttributes() - 1);
            eval = new Evaluation(data);
            Random rand = new Random(1); // using seed = 1
            int folds = 10;
            eval.crossValidateModel(classifier, data, folds, rand);
            sb.append("").append(eval.toClassDetailsString()).append("\n");
            sb.append("Instances: ").append(eval.numInstances()).append("\n");
            sb.append("Correct: ").append(eval.correct()).append("\n");
            sb.append("Incorrect: ").append(eval.incorrect()).append("\n");
            sb.append("pctCorrect: ").append(eval.pctCorrect()).append("\n");
            sb.append("pctIncorrect: ").append(eval.pctIncorrect()).append("\n");
            sb.append("Num Classes: ").append(data.numClasses()).append("\n");
            sb.append("Num atributos: ").append(data.numAttributes()).append("\n");
            sb.append("Micro:").append(getMicroAverage(eval, data)).append("\n");
            sb.append("Macro:").append(getMacroAverage(eval, data));
            System.out.println(sb.toString());
            salvaLinhaDados("baseline_" + args[0] + ".txt", sb.toString());
        } catch (Exception ex) {
            // Logger.getLogger(WekaSimulation.class.getName()).log(Level.SEVERE, null, ex);
        }

    }

    public static double getMacroAverage(Evaluation eval, Instances data) {
        double macroMeasure;
        double macroPrecision = 0;
        double macrorecall = 0;

        for (int i = 0; i < data.numClasses(); i++) {
            macroPrecision += eval.precision(i);
            macrorecall += eval.recall(i);
        }
        macroPrecision = macroPrecision / data.numClasses();
        macrorecall = macrorecall / data.numClasses();
        macroMeasure = (macroPrecision * macrorecall * 2) / (macroPrecision + macrorecall);
        System.out.println("macroMeasure: " + macroMeasure);

        return macroMeasure;
    }

    public static double getMicroAverage(Evaluation eval, Instances data) {
        double TP = 0;
        double TP_plus_FP = 0;
        double TP_plus_FN = 0;
        double microPrecision;
        double microRecall;
        double microMeasure;

        for (int i = 0; i < data.numClasses(); i++) {
            TP += eval.truePositiveRate(i);
            TP_plus_FP += eval.truePositiveRate(i) + eval.falsePositiveRate(i);
            TP_plus_FN += eval.truePositiveRate(i) + eval.falseNegativeRate(i);
        }
        microPrecision = TP / TP_plus_FP;
        microRecall = TP / TP_plus_FN;
        microMeasure = (microPrecision * microRecall * 2) / (microPrecision + microRecall);

        System.out.println("microMeasure: " + microMeasure);

        return microMeasure;
    }

    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    private static void salvaLinhaDados(String fileName, String dado) throws IOException {
        try (FileWriter fw = new FileWriter(fileName); BufferedWriter bw = new BufferedWriter(fw)) {
            bw.write(dado);
            bw.newLine();
            bw.close();
            fw.close();
        }
    }

}
