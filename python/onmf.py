def oNMF(X, minibatch):
	# X in the input data (rows are genes, columns are cells)
	# X usually has a shape of 20,000 x 5,000 (5000 columns is quite slow in python)
	# X is a scipy sparse CSC matrix
	# minibatch is a boolean (perform minibatchkmeans or regular kmeans)
	params = {'orthogonal_NMF' : {
			'minibatchkmeans' : True,
			'n_cells' : 0,
			'orthogonal' : [1,0],
			'n_iter' : 200,
			'verbose' : 1,
			'residual' : 1e-4,
			'tof' : 1e-4,
			'n_clusters' : 10,
			'cost_function_penalty' : 0.7,
			'kmeans' : {
				'init' : 'k-means++',
				'n_init' : 10,
				'max_iter' : 300,
				'tol' : 0.0001,
				'precompute_distances' : 'auto',
				'verbose' : 0,
				'random_state' : None,
				'copy_x' : True,
				'n_jobs' : -1,
				'algorithm' : 'auto'
			},
			'minibatchkmeans' : {
				'init' : 'k-means++',
				'max_iter' : 100,
				'batch_size' : 100,
				'verbose' : 0,
				'compute_labels' : True,
				'random_state' : None,
				'tol' : 0.0,
				'max_no_improvement' : 10,
				'init_size' : None,
				'n_init' : 3,
				'reassignment_ratio' : 0.01
			},
			'plot' : {
				'c_realdata' :'#000000',
				'c_generateddata' :'#8DD3C7',
				'c_centroids' :'magenta',
				'size_regular' : 2,
				'size_centroids' : 100,
				'densities' : False,
				'nfeats' : 5,
				'alpha' : 0.3
			},
		}
	}
	p = params['orthogonal_NMF']
	k = p['n_clusters']
	orthogonal=p['orthogonal']
	n_iter=p['n_iter']
	verbose=p['verbose']
	residual=p['residual']
	tof=p['tof'] #tolerance

	r, c = X.shape #r number of features(genes), c number of samples (cells)
	if minibatch == False:
		A, inx = kmeans(X.T)
	else:
		A, inx = minibatchkmeans(X.T, params)
	Y = ss.csc_matrix((np.ones(c), (inx, range(c))), shape=(k,c)).todense()
	Y = Y+0.2
	if np.sum(orthogonal) == 2:
		S = A.T.dot(X.dot(Y.T))
	else:
		S = np.eye(k)

	X=X.todense()
	XfitPrevious = np.inf
	for i in range(n_iter):
		if orthogonal[0]==1:
			A=np.multiply(A,(X.dot(Y.T.dot(S.T)))/(A.dot(A.T.dot(X.dot(Y.T.dot(S.T))))))
		else:
			A=np.multiply(A,(X.dot(Y.T))/(A.dot(Y.dot(Y.T))))
		A = np.nan_to_num(A)
		A = np.maximum(A,np.spacing(1))

		if orthogonal[1]==1:
			Y=np.multiply(Y,(S.T.dot(A.T.dot(X)))/(S.T.dot(A.T.dot(X.dot(Y.T.dot(Y))))))
		else:
			Y=np.multiply(Y,(A.T.dot(X))/(A.T.dot(A.dot(Y))))
		Y = np.nan_to_num(Y)
		Y = np.maximum(Y,np.spacing(1))

		if np.sum(orthogonal) == 2:
			S=np.multiply(S,(A.T.dot(X.dot(Y.T)))/(A.T.dot(A.dot(S.dot(Y.dot(Y.T))))))
			S=np.maximum(S,np.spacing(1))
		
		if np.mod(i,10) == 0 or i==n_iter-1:
			if verbose:
				print('......... Iteration #%d' % i)
			XfitThis = A.dot(S.dot(Y))
			fitRes = np.linalg.norm(XfitPrevious-XfitThis, ord='fro')
			XfitPrevious=XfitThis
			curRes = np.linalg.norm(X-XfitThis,ord='fro')
			if tof>=fitRes or residual>=curRes or i==n_iter-1:
				print('Orthogonal NMF performed with %d iterations\n' % i)
				break
	return A, Y