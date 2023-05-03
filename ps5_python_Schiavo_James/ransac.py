import numpy as np

#ransac: randomly picks some points to define your model
def ransac(matches,sample_size, outlier_p=0.5, biased=True, retries=None):
    (consensus_strength, cs) = (np.NINF, None)
    matches_n = len(matches)
    matches = np.asarray(matches)
    probs_sum = lambda x: 3 * x * x + x * (x + 1) / 2
    biased_prob = np.arange(4 * matches_n, 3 * matches_n, -1) / probs_sum(matches_n)
    if retries is None:
        #calculating N
        max_retries = int(np.log(0.01) / np.log(1 - outlier_p ** sample_size))
        max_retries = min(max_retries, 1000)
    else:
        max_retries = retries
    for i in range(max_retries):
        #randomly sample the number of points required to fit the model
        rand_sample = np.random.choice(matches_n, sample_size, p=biased_prob) if biased else np.random.choice(matches_n, sample_size)
        pts_a = np.asarray([match[0] for match in matches])
        pts_b = np.asarray([match[1] for match in matches])
        #consensus set = points within error bounds(distance threshold) of the model
        if sample_size == 1:
            print("Transformation!!!!!")
            i = rand_sample[0]
            trans_ij = pts_b[i] - pts_a[i]
            print("trans_ij: " + str(trans_ij))
            consensus_set = np.linalg.norm((pts_a + trans_ij) - pts_b, axis=1) < 5
            cs_strength_i = consensus_set.sum()
            print("cs_strength_i: " + str(cs_strength_i))
            print("percentage of matches: " + str((cs_strength_i/consensus_set.shape[0])*100))
            cs_i = consensus_set
        else:
            print("Similarity!!!!!")
            pts_a, pts_b = matches[:, 0, :], matches[:, 1, :]
            transformation_m = similarity_mat(pts_a[rand_sample], pts_b[rand_sample])
            print("transformation_m: " + str(transformation_m))
            pts_a_homo = np.append(pts_a, np.ones((pts_a.shape[0], 1)), axis=1)
            pts_b_est = np.matmul(transformation_m, pts_a_homo.T).T
            consensus_set = np.linalg.norm(pts_b_est - pts_b, axis=1) < 150
            #print("consensus_set: " + str(consensus_set))
            cs_strength_i = consensus_set.sum()
            print("cs_strength_i: " + str(cs_strength_i))
            print("percentage of matches: " + str((cs_strength_i/consensus_set.shape[0])*100))
            cs_i = consensus_set
        #return model with max consensus strength over N trials
        (consensus_strength, cs) = (max(consensus_strength, cs_strength_i), cs_i if consensus_strength < cs_strength_i else cs)
    return list(np.where(cs)[0])

def similarity_mat(pts_a, pts_b):
    l = min(pts_a.shape[0], pts_b.shape[0])
    a = np.zeros((2 * l, 4))
    b = np.zeros((2 * l,))
    for (u, v, *rest), (u_, v_, *rest), i in zip(pts_a, pts_b, range(l)):
        a[2 * i, :] = [u, -v, 1, 0]
        a[2 * i + 1, :] = [v, u, 1, 0]
        b[2 * i] = u_
        b[2 * i + 1] = v_
    m, res, _, _ = np.linalg.lstsq(a, b, rcond=None)
    a, b, c, d = m
    return np.asarray([[a, -b, c], [b, a, d]])




