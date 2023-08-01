# Deep Learning Project: Predicting Charity Funding for Alphabet Soup

## Introduction
In this project, I used deep learning and neural networks to predict whether funding applications submitted to Alphabet Soup would be successful. Alphabet Soup is a charitable organization that has funded over 34,000 other organizations.

## Data Processing
I removed irrelevant information from the dataset, such as EIN and NAME columns, as they didn't contribute to the model. The remaining columns were treated as features for the model. To handle high fluctuations, I replaced "CLASSIFICATION" and "APPLICATION_TYPE" with 'Other'. Additionally, I grouped rare categorical variables together into a new value, 'Other'.

## Modeling
For the model, I designed a three-layer neural network. The number of hidden nodes in each layer was determined by the number of features in the dataset. The initial model had 477 parameters.

## Training and Evaluation
In the first attempt, the model achieved a prediction accuracy of 72%, which fell short of the desired 75%.

## Optimization
To improve the model's performance, I reintroduced the "NAME" feature in the second attempt. This change led to a significant improvement, achieving an accuracy of 79%, exceeding the target by 4%. The model now had 3,298 parameters.

## Conclusion
In conclusion, my deep learning project successfully predicted the success of funding applications for Alphabet Soup. By processing the data carefully, implementing a three-layer neural network, and reintroducing the "NAME" feature, I achieved an accuracy of 79%. The model's performance could be further improved with ongoing exploration and optimization. This project highlights the potential of using deep learning to make informed decisions on funding applications for charitable organizations like Alphabet Soup.

 