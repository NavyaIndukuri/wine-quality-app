import org.apache.spark.sql.*;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.VectorAssembler;

public class ValidateApp {
    public static void main(String[] args) {
        // Step 1: Initialize Spark Session
        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Validation")
                .master("local[*]")
                .getOrCreate();

        try {
            // Step 2: Load Validation Dataset
            String validationPath = "file:///home/ubuntu/ValidationDataset.csv";
            Dataset<Row> validationData = spark.read()
                    .option("header", "true")
                    .option("delimiter", ";")
                    .option("inferschema", "true")
                    .csv(validationPath);

            // Step 3: Clean Column Names (Remove Double Quotes)
            for (String colName : validationData.columns()) {
                String cleanColName = colName.replaceAll("\"", "").trim(); // Remove all double quotes
                validationData = validationData.withColumnRenamed(colName, cleanColName);
            }

            System.out.println("Validation Dataset Loaded and Cleaned Successfully:");
            validationData.show(5);
            validationData.printSchema();

            // Step 4: Prepare Features for Model
            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(new String[]{
                            "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                            "pH", "sulphates", "alcohol"})
                    .setOutputCol("features");

            Dataset<Row> transformedValidationData = assembler.transform(validationData);

            // Step 5: Load the Saved Logistic Regression Model
            String modelPath = "file:///home/ubuntu/wine_quality_model";
            LogisticRegressionModel model = LogisticRegressionModel.load(modelPath);
            System.out.println("Model Loaded Successfully");

            // Step 6: Make Predictions
            Dataset<Row> predictions = model.transform(transformedValidationData);
            System.out.println("Predictions:");
            predictions.select("features", "prediction").show(10);

            // Step 7: Analyze Predictions (Optional)
            System.out.println("Prediction Summary:");
            predictions.groupBy("prediction").count().show();

            // Display Accuracy Metrics (if actual labels are present)
            if (validationData.columns().length > 0 && validationData.columns()[validationData.columns().length - 1].equals("quality")) {
                Dataset<Row> correctPredictions = predictions.filter("prediction == quality");
                long correctCount = correctPredictions.count();
                long totalCount = predictions.count();

                double accuracy = (double) correctCount / totalCount;
                System.out.printf("Model Accuracy: %.2f%%\n", accuracy * 100);
            }

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            // Step 8: Stop Spark Session
            spark.stop();
            System.out.println("Spark Session Stopped.");
        }
    }
}

