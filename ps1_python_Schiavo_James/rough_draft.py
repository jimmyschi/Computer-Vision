for it in range(iters):
        ssd = 0
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                minimum = min(X[:][j])
                print(minimum)
                maximum = max(X[:][j])
                print(maximum)
                target_range = maximum - minimum
                print(target_range)
                means[:][j] = rand.randint(0,target_range)
                #print(means)
                min_cluster = 1
                pdist = torch.nn.PairwiseDistance(p=2)
                dist = pdist(torch.from_numpy(X),torch.from_numpy(means))
                #for k in range(K):
                    #dist = pdist(torch.from_numpy(X[i][j]),torch.from_numpy(means[k][j]))
                    #if dist < min_dist: 
                       #min_dist = dist
                        #min_cluster = k
                #means[:][j] = np.mean()





                """"
                    for m in range(X.shape[1]):
                        print("ids: " + str(ids))
                        for n in range(ids.shape[0]):
                            print("k: " + str(k))
                            if ids[n] == (k + 1):
                                sum = sum + means[k][m]
                                count = count + 1
                        print("sum: " + str(sum))
                        means[k][m] = sum/count
    ssd = (sum(np.min(cdist(X, means,'euclidean'), axis=1))**2) 
    """            