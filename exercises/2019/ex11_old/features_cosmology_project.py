
import os
import os.path
import random
import csv
import sys

import PIL.Image
import numpy as np
import sklearn.linear_model as sklm
import sklearn.model_selection as skms
import sklearn.preprocessing as skpp
import sklearn.metrics as skmet

def csv_to_dict(csv_path):
    with open(csv_path,'r') as fp:
        csv_fp=csv.reader(fp)
        next(csv_fp)
        d = dict(filter(None, csv_fp))
        return d

def extract_feats(img_arr):
    hist,_=np.histogram(img_arr,bins=10)

    # Consider more sophisticated features here: Frequency domain energy, ROI histograms, shape descriptors, etc...
    
    return hist

if __name__=="__main__":

    try:
        data_path=os.environ["COSMOLOGY_DATA"].strip()
    except KeyError:
        print("ERROR: Provide data path via environment...")
        sys.exit(1)

    # Parameters
    feat_size=10
    train_ratio=0.7

    # Paths
    labeled_path=os.path.join(data_path,"labeled")
    label_file=os.path.join(data_path,"labeled.csv")

    # Initialization
    label_dict=csv_to_dict(label_file)
    img_prefixes=list(label_dict.keys())
    random.shuffle(img_prefixes)
    n_train=int(train_ratio*len(img_prefixes))
    n_test=len(img_prefixes)-n_train
    train_mat=np.zeros((n_train,feat_size))
    train_y=np.zeros(n_train)
    test_mat=np.zeros((n_test,feat_size))
    test_y=np.zeros(n_test)
    train_idx=0
    test_idx=0

    # Assemble train/test feature matrices / label vectors
    for idx,img_prefix in enumerate(img_prefixes):

        print("Image: {}/{}".format(idx+1,len(img_prefixes)))
        raw_image=PIL.Image.open(os.path.join(labeled_path,"{}.png".format(img_prefix)))
        img_arr=np.array(raw_image.getdata()).reshape(raw_image.size[0],raw_image.size[1]).astype(np.uint8)
        img_feats=extract_feats(img_arr)
        label=float(label_dict[img_prefix])

        if idx<n_train:
            train_mat[train_idx,:]=img_feats
            train_y[train_idx]=label
            train_idx+=1
        else:
            test_mat[test_idx,:]=img_feats
            test_y[test_idx]=label
            test_idx+=1

    # Consider saving features/labels to disk here...

    # Try more rigorous evaluation like replications or CV splits here...

    # Normalize feature matrices
    std_model=skpp.StandardScaler()
    train_mat=std_model.fit_transform(train_mat)
    test_mat=std_model.transform(test_mat)

    # Fit logistic regression model
    base_model=sklm.SGDClassifier(loss="log",class_weight="balanced")

    # Consider non-linear methods like Random Forest here...
    
    ml_model=skms.GridSearchCV(base_model,{"alpha": [0.0001,0.001,0.01,0.1,1.0]},verbose=5)
    ml_model.fit(train_mat,train_y)
    pred_test_y=ml_model.predict(test_mat)

    # Classification diagnostics
    scores_test_y=ml_model.predict_proba(test_mat)[:,1]
    print("Held out accuracy: {:.3f}".format(skmet.accuracy_score(test_y,pred_test_y)))
    print("Held out AUROC: {:.3f}".format(skmet.roc_auc_score(test_y,scores_test_y)))
    print("Held out AUPRC: {:.3f}".format(skmet.average_precision_score(test_y,scores_test_y)))
