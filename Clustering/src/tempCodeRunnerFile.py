 KMeans(n_clusters=10, random_state = 0)
    kmeans.fit(all_training_imgs)
    model_labels = kmeans.predict(all_training_imgs)
    model_labels = get_actual_labels(all_training_labels, model_labels, cluster_num=10, cluster_size=892)
    print("model labels for the first 100 features: ", model_labels[: 100])
    score = accuracy_score(all_training_labels, model_labels)
    print("