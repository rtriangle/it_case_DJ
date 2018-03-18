from sklearn.preprocessing import Imputer


def fill_nan(data, mode):
    """
    mode: ['mean', 'median', 'linear', 'time', 'index', 'values', 'nearest', 'zero',
          'slinear', 'quadratic', 'cubic', 'barycentric', 'krogh',
          'polynomial', 'spline', 'piecewise_polynomial',
          'from_derivatives', 'pchip', 'akima']
    """
    if mode in ['mean', 'median']: 
        columns = data.columns
        imputer = Imputer(missing_values='NaN', strategy=mode, axis=0)
        return pd.DataFrame(imputer.fit_transform(data), columns=columns)
    else:
        return data.interpolate(method=mode)


def add_features(data, modes=['product'], coll_names=[], coll_pair_names=[], 
                      include_same_column=False, 
                      include_different_columns=True):
    """
    mode: ['product', 'sqrt', 'log']
    """
    if coll_pair_names == []:
        coll_pair_names = data.columns()
    if coll_names == []:
        coll_names = data.columns()
    for col1 in coll_pair_names:
        for col2 in coll_pair_names:
            if col1 == col2 and include_same_column or \
               col1 != col2 and include_different_columns:
                if 'product' in modes:
                    data[col1 + '*' + col2] = data[col1] * data[col2]
    for col in coll_names:
        if 'sqrt' in modes:
            data['sqrt_' + col] = np.sqrt(data[col])
        if 'log' in modes:
            data['log_' + col] = np.log(data[col])
    return data


def cut_by_percentile(data, percentile=95, columns=[]):
    if columns == []:
        columns = data.columns()
    for col in columns:
        threshold = np.percentile(data[col], percentile)
        data[col] = np.array(list(map(lambda element: min(threshold, element), data[col])))
    return data


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
def select_top_k_features(X, y, k=10):
    columns = ['FEATURE' + str(i) for i in range(k)]
    return pd.DataFrame(SelectKBest(f_classif, k=k).fit_transform(X, y), columns=columns)
