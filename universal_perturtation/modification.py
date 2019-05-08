 
 
 def deepfool(image, f, target, grads, num_classes=10, overshoot=0.02, max_iter=np.inf):

    """
       :param image: Image of size HxWx3
       :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
       :param grads: gradient functions with respect to input (as many gradients as classes).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 10)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    if target < 0 or target >= num_classes:
		raise ValueError('wrong target')
	else:
	    t = target
    f_image = np.array(f(image)).flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1] #x.argsort()将x中的元素从小到大排列并提出其索引输出，[::-1]表示从头到尾将数组反转

    I = I[0:num_classes]
    label = I[0]    #此时I[0]为f(image)取值最大的，即image原来的label
    
	print('label= %d' %label)
	if t == label:
	    raise ValueError('useless target')
    input_shape = image.shape
    pert_image = image

    f_i = np.array(f(pert_image)).flatten()
    k_i = int(np.argmax(f_i))   #f(image)最大元素对应的索引值

    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)
	loop_i = 0
	
	for k in range(num_classes):
		if I[k] == t:
			t = k
    print('t=k= %d' %t)
		
	while k_i != t and loop_i < max_iter:

			pert = np.inf
			gradients = np.asarray(grads(pert_image,I))   #np.asarray(a, dtype=None, order=None)将结构数据转化为ndarray，与np.array不同的是（默认情况下）将会copy该对象，而 np.asarray 除非必要，否则不会copy该对象。
			# set new w_k and new f_k
			w_t = gradients[t, :, :, :, :] - gradients[0, :, :, :, :]   #更新w_k
			f_t = f_i[I[t]] - f_i[I[0]]     #更新f_k
			pert_t = abs(f_t)/np.linalg.norm(w_t.flatten())
			# compute r_i and r_tot
			r_i =  pert_t * w_t / np.linalg.norm(w_t)
			r_tot = r_tot + r_i

			# compute new perturbed image
			pert_image = image + (1+overshoot)*r_tot
			loop_i += 1

			# compute new label
			f_i = np.array(f(pert_image)).flatten()
			k_i = int(np.argmax(f_i))

	 r_tot = (1+overshoot)*r_tot

	 return r_tot, loop_i, k_i, pert_image