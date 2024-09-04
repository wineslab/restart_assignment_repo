from functions import *

mpl.use('macosx')

# Column headers
columns_list = [
    "Timestamp", "IMSI", "slice_id", "slice_prb", "scheduling_policy", "dl_mcs",
    "dl_n_samples", "dl_buffer [bytes]", "tx_brate downlink [Mbps]", "tx_pkts downlink",
    "dl_cqi", "ul_mcs", "ul_n_samples", "ul_buffer [bytes]", "rx_brate uplink [Mbps]",
    "rx_pkts uplink", "rx_errors uplink (%)", "ul_sinr", "sum_requested_prbs", "sum_granted_prbs"
]

# Dataset filenames
dataset_filenames = {
    "training": "Data/dataset_restart_training.pkl",
    "testing": "Data/dataset_restart_testing.pkl",
}

rs = 42


# Main execution logic
def main():


    # Configurations
    dataset_filename = dataset_filenames["training"]
    
    # Load Train dataset
    dataset = load_dataset(dataset_filename)
    
    # Ensure the Timestamp column is in datetime format
    dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'], errors='coerce')

    # Load Test Dataset

    test_filename = 'dataset_restart_testing_fake.pkl'
    test_dataset = pd.read_pickle(test_filename)
    
    # Ensure the Timestamp column is in datetime format
    test_dataset['Timestamp'] = pd.to_datetime(test_dataset['Timestamp'], errors='coerce')

    # Prepare Test Data for Performance Evaluation

    #Training
    
    X = dataset.drop(["Timestamp", "IMSI", "slice_id"], axis=1)
    y =  dataset.loc[:, 'slice_id']
    
    #Test
    X_test = test_dataset.drop(["Timestamp", "IMSI", "slice_id"], axis=1)
    y_test =  test_dataset.loc[:, 'slice_id']
    
    #Normalize
    X_norm, X_test_norm, stats_test = normalize_dataset(X, X_test)
    
    
    ### UN-SUPERVISED (CLUSTERING via k-means)
    print('UNSUPERVISED LEARNING: CLUSTERING')
    
    ###### MODIFY THIS SECTION USING AD-HOC K FOR YOUR CUSTOM CLUSTERING CONFIGURATION

    k_test =[] #select best k as input hyper-parameter for clustering on test set

    for k in k_test:
        
        kmeans_model = KMeans(n_clusters=k, init="k-means++", n_init='auto')
    
    
        kmeans_model.fit(X_norm) # train the model
    
        cluster_labels = kmeans_model.predict(X_test_norm) # assign label to test data
    
        compute_unsupervised_performance(k, y_test, X_test_norm, cluster_labels, ns=1000)

    print('')

    ### SUPERVISED (CLASSIFICATION)

    print('SUPERVISED LEARNING: CLASSIFICATION')


    ###### MODIFY THIS SECTION INTRODUCING YOUR CUSTOM CLASSIFIER(S)

    # Put here your final Classifier(s) with Hyper-Parameters tuned
    classifiers = {
        "Linear Regression": RidgeClassifier(fit_intercept=True, solver='svd')
        }


    for i, (clf_name, clf) in enumerate(classifiers.items()):
        
        clf.fit(X_norm, y) #train
        
        output = clf.predict(X_test_norm) #predict
        
        f1 = f1_score(y_test, output, average='micro')
        acc = accuracy_score(y_test, output)
        print_performance_supervised(clf_name, acc, f1, y_test, output)


if __name__ == "__main__":
    main()
