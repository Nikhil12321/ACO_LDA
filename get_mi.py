import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif as mic
from sklearn.feature_selection import mutual_info_regression as mir
import gcmi


def get_mutual_information(filename):
    data = pd.read_csv(filename, dtype=None)

    headers_list = data.columns.get_values().tolist()
    X = data[headers_list[:-1]]
    y = data[headers_list[-1]]
    #Mutual information between each feature and class. Use mi_fc[i].
    mi_fc = mic(X, y, discrete_features = False)

    #Mutual information between each feature pair
    #To get entropy of a feature, use mi_ff[i][i]
    mi_ff = []
    for i in headers_list[:-1]:
        feature = data[i]
        mi_result_feature = mir(X, feature, discrete_features = False)
        mi_ff.append(mi_result_feature)

    #cmi_ffc contains Conditional Mutual Information of fi,fs:C. Use cmi_ffc[feature_i][feature_s]
    cmi_ffc = []
    for i in xrange(0, len(headers_list)-1):
        cmi = []
        for j in xrange(0, len(headers_list)-1):
            if i == j:
                cmi.append(0)
            else:
                string1 = headers_list[i]
                string2 = headers_list[j]
                f1_c = data[string1].as_matrix()
                f2_c = data[string2].as_matrix()
                f3_d = data.defects.astype(int).as_matrix()
                result = gcmi.gccmi_ccd(f1_c, f2_c, f3_d, 2)
                cmi.append(result[0])
        cmi_ffc.append(cmi)
    return mi_fc, mi_ff, cmi_ffc
