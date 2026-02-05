# Model Card

## Model Details
This model is a supervised binary classification model trained to predict whether an individual earns more than $50,000 per year based on U.S. Census demographic data. The model is implemented using scikit-learn’s Logistic Regression algorithm. Categorical features are processed using one-hot encoding, and the target variable is binarized prior to training.

The model was trained using a train-test split approach and saved for later inference and deployment through a REST API.

## Intended Use
The intended use of this model is educational and demonstrative. It is designed to showcase how a machine learning pipeline can be built, trained, evaluated, and deployed using FastAPI. The model may be used to explore patterns in census income data but should not be used for real-world decision-making, such as hiring, lending, or policy enforcement.

## Training Data
The training data comes from the U.S. Census Bureau Adult Income dataset. The dataset contains demographic and employment-related features such as age, education, workclass, marital status, occupation, race, sex, hours worked per week, and native country. The target variable indicates whether an individual’s income exceeds $50,000 per year.

The dataset includes both numerical and categorical variables. Categorical features were encoded using one-hot encoding, and the dataset was split into training and testing subsets prior to model training.

## Evaluation Data
The evaluation data consists of a held-out test subset created using a train-test split from the original dataset. This data was not used during training and was processed using the same preprocessing pipeline and encoders as the training data.

## Metrics
The model was evaluated using the following classification metrics:
- Precision
- Recall
- F1 Score (F-beta with beta = 1)

On the test dataset, the model achieved approximately:
- Precision: 0.73
- Recall: 0.56
- F1 Score: 0.64

In addition to overall performance, the model’s performance was evaluated across slices of the data based on categorical features. Metrics for each slice were computed and stored in `slice_output.txt` to assess whether performance varied across different demographic groups.

## Ethical Considerations
This model is trained on demographic data that includes sensitive attributes such as race, sex, and marital status. As a result, the model may reflect historical biases present in the dataset. Predictions made by this model could disproportionately impact certain groups if used in real-world decision-making contexts.

Care should be taken to avoid deploying this model in scenarios where predictions could lead to discriminatory outcomes. Any real-world use would require extensive bias analysis, fairness evaluation, and human oversight.

## Caveats and Recommendations
This model is a simplified baseline classifier and is not optimized for production use. It does not account for concept drift, changing economic conditions, or evolving demographic trends. The model was trained using a limited feature set and basic preprocessing techniques.

Future improvements could include:
- Hyperparameter tuning
- Feature scaling or selection
- Alternative model architectures
- More robust bias and fairness evaluation

The model should be retrained and reevaluated before any deployment beyond demonstration or learning purposes.
