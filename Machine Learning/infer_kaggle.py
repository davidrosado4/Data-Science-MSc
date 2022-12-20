from xgboost import XGBClassifier
import pandas as pd

from sklearn.utils import resample


def predict(x_train, y_train, x_test):
    clf = XGBClassifier(
        learning_rate=0.01,
        min_child_weight=1,
        max_depth=40,
        n_estimators=300,
        gamma=5,
        subsample=1.0,
        colsample_bytree=0.8
        )

    clf.fit(x_train, y_train)
    return clf.predict(x_test)


def main():
    x_train = pd.read_csv('nd_clear_train.csv', index_col = 0)
    y_train = pd.read_csv('fraudulent.csv', index_col = 0)
    x_test = pd.read_csv('nd_clear_test.csv', index_col = 0)
    
    # concatenate our training data back together
    auxX = pd.concat([x_train, y_train], axis=1)

    # separate minority and majority classes
    not_fraud = auxX[auxX['fraudulent']==0]
    fraud = auxX[auxX['fraudulent']==1]

    # upsample minority
    fraud_upsampled = resample(fraud,
                              replace=True, # sample with replacement
                              n_samples=len(not_fraud), # match number in majority class
                              random_state=27) # reproducible results

    # combine majority and upsampled minority
    upsampled = pd.concat([not_fraud, fraud_upsampled])

    y_res = upsampled['fraudulent']
    X_res = upsampled.drop('fraudulent', axis=1)    
    
    y_hat = predict(X_res, y_res, x_test)
    

    # process the output according to the submission style
    y_hat = pd.DataFrame(y_hat)
    print(y_hat)
    y_hat.index.name = 'Id'
    print(y_hat)
    y_hat.columns = ['Category']
    print(y_hat)
    y_hat.to_csv('y_hat.csv')


if __name__ == '__main__':
    main()
