from sklearn.model_selection import train_test_split

def split(X, y, test_size):
    return train_test_split(X, y, test_size=test_size, stratify=y)
    # test_size is the fraction of testing data
