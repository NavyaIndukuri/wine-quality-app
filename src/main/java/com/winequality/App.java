import org.apache.spark.sql.*;
import org.apache.spark.sql.types.*;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

public class App {
    public static void main(String[] args) {
        // Step 1: Create a Spark Session
        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Prediction")
                .master("local[*]")
                .getOrCreate();

        try {
            // Step 2: Define the Schema for Dataset
            StructType schema = new StructType()
                    .add("fixed_acidity", "double")
                    .add("volatile_acidity", "double")
                    .add("citric_acid", "double")
                    .add("residual_sugar", "double")
                    .add("chlorides", "double")
                    .add("free_sulfur_dioxide", "double")
                    .add("total_sulfur_dioxide", "double")
                    .add("density", "double")
                    .add("pH", "double")
                    .add("sulphates", "double")
                    .add("alcohol", "double")
                    .add("quality", "double");

            // Step 3: Load the Dataset
            String dataPath = "file:///home/ubuntu/TrainingDataset.csv";
            Dataset<Row> data = spark.read()
                    .option("header", "true")
                    .option("delimiter", ";")
                    .option("inferschema", "true")
                    .schema(schema)
                    .csv(dataPath);

            System.out.println("Dataset Loaded Successfully:");
            data.show(5);

            // Step 4: Prepare Features using VectorAssembler
            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(new String[]{
                            "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                            "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
                            "pH", "sulphates", "alcohol"})
                    .setOutputCol("features");

            Dataset<Row> transformedData = assembler.transform(data)
                    .withColumnRenamed("quality", "label"); // Rename quality to label

            System.out.println("Transformed Data:");
            transformedData.select("features", "label").show(5);

            // Step 5: Split the Data into Training and Test Sets
            Dataset<Row>[] splits = transformedData.randomSplit(new double[]{0.8, 0.2}, 1234L);
            Dataset<Row> trainingData = splits[0];
            Dataset<Row> testData = splits[1];

            // Step 6: Train the Logistic Regression Model
            LogisticRegression logisticRegression = new LogisticRegression();
            LogisticRegressionModel model = logisticRegression.fit(trainingData);

            System.out.println("Model Training Completed");

            // Save the Model
            model.save("file:///home/ubuntu/wine_quality_model");
            System.out.println("Model Saved Successfully");

            // Step 7: Make Predictions
            Dataset<Row> predictions = model.transform(testData);
            predictions.select("features", "label", "prediction").show(10);

            // Step 8: Evaluate the Model
            MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                    .setLabelCol("label")
                    .setPredictionCol("prediction")
                    .setMetricName("f1");

            double f1Score = evaluator.evaluate(predictions);
            System.out.println("F1 Score: " + f1Score);

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            spark.stop();
            System.out.println("Spark Session Stopped.");
        }
    }
}

