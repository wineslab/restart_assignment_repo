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
    "complete": "Data/dataset_restart_complete.pkl"
}

rs = 42 #random_state

supervised = False # set to True to perform hp tuning + prediction via supervised learning
unsupervised = False # set to True to use unspervised learning
k = None # Set to None for tuning of k, otherwise to int greater than 1 to perform clustering.
pca_flag = False # If k is None, set to True to evaluate the benefit of PCA on the tuning of k

# Main execution logic
def main():


    # Configurations
    dataset_filename = dataset_filenames["training"]
    is_split_dataset_active = False
    extract_rate = 0.05

    # Load dataset
    dataset = load_dataset(dataset_filename)

    # Split dataset if flag is active
    if is_split_dataset_active:
        split_and_save_dataset(dataset, extract_rate, dataset_filenames)

    # Ensure the Timestamp column is in datetime format
    dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'], errors='coerce')


    # CLASSIFICATION PIPELINE
    cross_val_k = 3


    #Comment above and use below to train/test on full dataset

    # Prepare Data for Training and Validation Evaluation
    X, X_test, y, y_test = train_test_split(dataset.drop(["Timestamp", "IMSI", "slice_id"], axis=1),
                                                        dataset.loc[:, 'slice_id'],
                                                        test_size=0.2, random_state=rs)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                      random_state=rs)  # 0.25
    
    X_train_norm, X_val_norm, stats_val = normalize_dataset(X_train, X_val)
    X_norm, X_test_norm, stats_test = normalize_dataset(X, X_test)


    if supervised:

        ### SUPERVISED

        # Select Classsifiers

        classifiers = {
            "Linear Regression": RidgeClassifier(solver='svd'),
        }


        # Select Hyper-Parameters
        params = {'Linear Regression': []} #choose parameters to optimize

        # Example to predict with plain logistic regression

        for i, (clf_name, clf) in enumerate(classifiers.items()):
          
          print(f'Train classifier on training set...')
          clf.fit(X_norm, y)
        
          print(f'Perform prediction on test set...')
          output = clf.predict(X_test_norm)
        
          save_predictions_supervised(clf_name, output)
  
        for i, (clf_name, clf) in enumerate(classifiers.items()):

            print(10 * '-')
        
            # Validation
        
            ###### 
            # Put Validation Logic Here
        
            ###### 
        
            # Testing
        
            ###### 
            # Put Testing Logic Here
        
            ###### 
        
            # save predictions: save_predictions_supervised(clf_name, output)

    if unsupervised:


        ### UNSUPERVISED

        if k is None:


            # Use function kmeans_helbow to select best k
            min_cl_km = 2
            max_cl_km = 8



            # Check how Silhouette Score varies with k with function kmeans_silhouette

            # k_silhouette = 
            
            # print(f"Best K Silhouette: {k_silhouette}") # extract k with best sil coeff

        else:
            
            # Once that k is tuned, use it to perform clustering and generate labels on Test Set

            ###### 
            # Put Clustering Logic Here
        
            ###### 
        
            # save_predictions_supervised(k, output)

            if pca_flag:

                ### PERFORM PCA to check how clusters look like in PC plane

                # Defining the number of principal components to generate
                n = min(X_norm.shape[0], X_norm.shape[1])  # get maximum n of components accepted by scikit.PCA
                
                
                # Finding principal components for the data
                pca = PCA(n_components=n, random_state=42)
                X_norm_pca = pd.DataFrame(pca.fit_transform(X_norm))
                
                # Get percentages of variance explained by each principal component
                # exp_var = 
                
                # Visualize the Cumulative Sum of Explained Variance 
                plt.figure(figsize=(10, 10))
                
                # find the least number of components that can explain more than x% variance
                xvar = 90
                
                
                # Make a scatter PLot of 1st vs 2nd components
                
                
                # Make a scatter PLot of 1st vs 2nd components, where data points are labelled according to the associated cluster
                # NB: also the centroids of the produced clustering configuration can be projected on the Principal plane 
                # applying the function transform() to the trained pca algorithm, giving as input the centroids 



plt.show(block=False)


if __name__ == "__main__":
    main()
