from libc.stdlib cimport malloc, free
from prop cimport A2prop, Channel

cdef class A2Prop:
	cdef A2prop c_a2prop

	def __cinit__(self):
		self.c_a2prop = A2prop()

	def load(self, str dataset, unsigned int m, unsigned int n, unsigned int seed):
		self.c_a2prop.load(dataset.encode(), m, n, seed)

	def compute(self, unsigned int nchn, chns, np.ndarray feat):
		cdef:
			Channel* c_chns = <Channel*> malloc(nchn * sizeof(Channel))
			float res1, res2
		res2 = 0.0
		for i in range(nchn):
			c_chns[i].type = chns[i]['type']
			c_chns[i].is_thr = (chns[i]['type'] > 1)
			c_chns[i].is_acc = (chns[i]['type'] % 2 == 1)

			c_chns[i].hop = chns[i]['hop']
			c_chns[i].dim = chns[i]['dim']
			c_chns[i].delta = chns[i]['delta']
			c_chns[i].alpha = chns[i]['alpha']
			c_chns[i].rra = chns[i]['rra']
			c_chns[i].rrb = chns[i]['rrb']

		res1 = self.c_a2prop.compute(nchn, c_chns, Map[MatrixXf](feat), res2)
		free(c_chns)
		return res1, res2
