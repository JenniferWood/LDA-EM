import doc,math,copy
import numpy as np
from scipy.special import gamma as gafunc
from scipy.special import digamma,polygamma,gammaln
import random

d = doc.Documents()
d.readDocs()

class LDA:

	def __init__(self,k):
		self.K = k
		self.rho = 0.000001
		self.rhoe = 0.000003
		self.rhoem = 0.0001

		self.docs = d.docs
		self.wd = d.wd

		self.alpha = [50.0 / self.K] * self.K
		self.beta = [1.0/len(self.wd)] * len(self.wd)

		self.maxL = float('-inf')

	def GibbsSammpling(self):
		#initial
		z = []
		ndk = [[0] * self.K] * len(self.docs)
		nd = [0] * len(self.docs)
		nkv = [[0] * len(self.wd)] * self.K
		nk = [0] * self.K

		for d in range(len(self.docs)):
			zd =[]
			for n in range(len(self.docs[d])):
				v = self.docs[d][n]
				zdn = random.randint(0,self.K-1)
				zd.append(zdn)
				ndk[d][zdn] += 1
				nd[d] += 1
				nkv[zdn][v] += 1
				nk [zdn] += 1
			z.append(zd)

		#inference
		t = 0
		while t<100:
			print 'iteration',t
			for d in range(len(self.docs)):
				for n in range(len(self.docs[d])):
					p = [0.0] * self.K
					v = self.docs[d][n]
					zdn = z[d][n]
				
					ndk[d][zdn] -= 1
					nd[d] -= 1
					nkv[zdn][v] -= 1
					nk[zdn] -= 1


					for k in range(self.K):
						p[k] = (ndk[d][k]+self.alpha[k])/(nd[d]+sum(self.alpha)) * (nkv[k][v] + self.beta[v])/(nk[k] + sum(self.beta))			

					p = self.normalize(p)
					#print p
					
					zdn = self.samplefrom(p)
					print "new z:",zdn
					#print zdn

					z[d][n] = zdn

					ndk[d][zdn]+=1
					nd[d] += 1
					nkv[zdn][v]+=1
					nk[zdn]+=1

			t+=1
		print z
	
	def samplefrom(self,p):
		i=1
		while i<len(p):
			p[i] += p[i-1]
			i+=1

		#print p
		sp = random.uniform(0,1)
		#print sp

		x = 0
		for pi in p:
			if pi > sp: return x
			x += 1

		if x==len(p): x-=1
		return x

	def EM(self):
		iteration = 0

		while  True:
			print "iteration",iteration
			self.Phi = [[[float(1)/self.K] * self.K]*len(doc) for doc in self.docs]
			self.Gamma = [[ak+float(len(doc))/self.K for ak in self.alpha] for doc in self.docs]
			self.Lambda = [[bi+float(self.K)/len(self.beta) for bi in self.beta]]*self.K

			#E-STEP

			print "############# E-STEP ################"
			et = 0
			while  True:
				print "trying the %d times" % et,
				phi = copy.deepcopy(self.Phi)
				gamma = copy.deepcopy(self.Gamma)
				lamda = copy.deepcopy(self.Lambda)


				for d in range(len(self.docs)):
					lbias = [0 for a in self.alpha]
					for n in range(len(self.docs[d])):
						for k in range(self.K):
							i = self.docs[d][n]
							self.Phi[d][n][k] = self.beta[i] * math.exp(digamma(self.Gamma[d][k]) - digamma(sum(self.Gamma[d])) \
								+ digamma(self.Lambda[k][i]) - digamma(sum(self.Lambda[k])))
						self.Phi[d][n] = self.normalize(self.Phi[d][n]) # Update Phi
						lbias = map(lambda (a,b):a+b, zip(lbias,self.Phi[d][n]))

					for k in range(self.K):
						self.Gamma[d][k] = self.alpha[k] + lbias[k]
					

				#Update Lambda
				for k in range(self.K):
					for i in range(len(self.wd)):
						self.Lambda[k][i] = self.beta[i]
						for d in range(len(self.docs)):
							for n in range(len(self.docs[d])):
								if self.docs[d][n] == i:
									self.Lambda[k][i] += self.Phi[d][n][k]

				edis = self.distance([phi,gamma,lamda],[self.Phi,self.Gamma,self.Lambda])
				print ":",edis
				if  edis < self.rhoe:
					break

				et +=1

			print "new Phi:",self.Phi[0][0]
			print "new Gamma:",self.Gamma[0]
			print "new Lambda:",self.Lambda[0][0:10]
			print ""
			print "############# M-STEP ################"
			#M-STEP
			altmp = copy.deepcopy(self.alpha)
			betmp = copy.deepcopy(self.beta)
			
			#update alpha
			self.Newton(self.alpha)
			#update beta
			self.Newton(self.beta)

			#self.savemodel(self.alpha)
			out = file('model/lda.alpha','w')
			for alphai in self.alpha:
				out.write('%f\t' % alphai)
			out.write("\n")

			#self.savemodel(self.beta)
			out = file('model/lda.beta','w')
			for betai in self.beta:
				out.write('%f\t' % betai)
			out.write("\n")

			out = file('model/lda.gamma','w')
			for gdoc in self.Gamma:
				for gw in gdoc:
					out.write('%f\t' % gw)
				out.write("\n")
			
			out = file('model/lda.lambda','w')
			for ltopic in self.Lambda:
				for lw in ltopic:
					out.write('%f\t' % lw)
				out.write("\n")

			'''
			out = file('model/lda.phi','w')
			for pdoc in self.Phi:
				for pdn in pdoc:
					for pk in pdn:
						out.write('%f\t' % pk)
					out.write("\n")
			'''

			if self.converged() is True:
				print "Model Converged"
				break
			
			iteration += 1

		

	def Newton(self,vec):
		if len(vec) == self.K:
			print "1, updating alpha------------------"
			ratio = len(self.docs)

		else:
			print "2, updating beta-------------------"
			ratio = self.K

		
		veck = copy.deepcopy(vec)
		
		t=0
		while True:
			print "updating the %d times" % t
			#print "x%d"%t,veck[0:10]

			gk = self.grad(veck)
			#print "gk%d"%t,gk[0:10]
			if self.normof(gk) < self.rho:
				print "after udating:",veck[0:10]
				print ""

				if len(vec) == self.K:
					self.alpha = veck
				else:
					self.beta = veck
				return

			Hk = [[ratio*polygamma(1,sum(veck))]*len(veck)]*len(veck)
			duijiao = [ratio*polygamma(1,vecki) for vecki in veck]
			Hk = np.mat(Hk) - np.mat(np.diag(duijiao))
			#print "Hk%d"%t,Hk[0]

			pk = (-1*(Hk.I)*(np.mat(gk).T)).T.tolist()[0]
			#print "pk%d"%t,pk[0:10]

			for i in range(len(veck)):
				veck[i] += pk[i]
			t += 1
			

	def L(self,vec,type):
		result = 0
		if type == 0:
			for d in range(len(self.docs)):
				result += math.log(gafunc(sum(vec)) - sum([math.log(gafunc(a)) for a in vec]))
				for k in range(self.K):
					result += (vec[k]-1)*(digamma(self.Gamma[d][k] - digamma(sum(self.Gamma[d]))))
		else:
			result = self.K * math.log(gafunc(sum(vec))) - self.K * sum([math.log(gafunc(a)) for a in vec])
			for k in range(self.K):
				for i in range(len(self.wd)):
					result+= (vec[i]-1)*(digamma(self.Lambda[k][i] - digamma(sum(self.Lambda[k]))))
		print "L",result
		return result

    # It's actually math.pow(realnorm,2)
	def normof(self,vec):
		norm2 = sum([v*v for v in vec])
		#print "norm2",norm2
		return norm2

	def grad(self,vec):
		if len(vec) == self.K:
			ratio = len(self.docs)
			bias = self.Gamma

		else:
			ratio = self.K
			bias = self.Lambda

		ds = digamma(sum(vec))
		grads = [ratio*ds-ratio*digamma(veci) for veci in vec]
		for i in range(len(vec)):
			for j in range(ratio):
				grads[i] += digamma(bias[j][i])-digamma(sum(bias[j]))
		
		#print "grads",grads[0:10]
		return grads
		

	def normalize(self,p):
		#print "before normalize",p
		s = sum(p)
		p = [pi/s for pi in p]
		#print "after normalize",p
		return p

	def Dig(self,x):
		if x == 0: return 0
		if x<0:
			x = -1*x
			return self.Dig(x)+math.pi*math.cos(x*math.pi)+1/x
		if x == 1: return -0.57721566490153286
		
		z = int(x)
		d = x-z
		if d == 0: 
			value = -0.57721566490153286
			d = 1
			z -= 1
		else:
			d2 = d*d
			d3 = d2*d
			value = -0.515095835950807+1.389927456533864*d-1.008336779674558/d+0.000557958765350/d2-0.586786525683560*d2-0.000014382050162/d3+0.142984009331572*d3

		i=0
		while i<z:
			value += float(1)/d
			d+=1
			i+=1 

		return value

	def Ddig(self,x):
		if x==0: return 0
		return float(1)/x + float(1)/(2*x*x)

	def distance(self,listA,listB):
		result = 0.0
		for i in range(len(listA)):
			if isinstance(listA[i],list):
				result += self.distance(listA[i],listB[i])
			else:
				result += math.pow(listB[i]-listA[i],2)
		return result

	def converged(self):
		E = self.K*gammaln(sum(self.beta))-self.K*sum([gammaln(betai) for betai in self.beta])
		E += gammaln(sum(self.alpha))-sum([gammaln(alphak) for alphak in self.alpha])
		for k in range(self.K):
			E -= gammaln(sum(self.Lambda[k]))
			E += sum([gammaln(self.Lambda[k][i])+(self.beta[i]-self.Lambda[k][i])*(digamma(self.Lambda[k][i]) - digamma(sum(self.Lambda[k]))) for i in range(len(self.wd))])
		for d in range(len(self.docs)):
			E -= gammaln(sum(self.Gamma[d]))
			E += sum([gammaln(self.Gamma[d][k])+(self.alpha[k]-self.Gamma[d][k])*(digamma(self.Gamma[d][k])-digamma(sum(self.Gamma[d]))) for k in range(self.K)])
			for n in range(len(self.docs[d])):
				for k in range(self.K):
					E += self.Phi[d][n][k]*(digamma(self.Gamma[d][k])-digamma(sum(self.Gamma[d]))-math.log(self.Phi[d][n][k]))
					i = self.docs[d][n]
					E += self.Phi[d][n][k]*(digamma(self.Lambda[k][i])-digamma(sum(self.Lambda[k])))

		print "L NOW IS",E
		if E <= self.maxL:
			return True

		else:
			self.maxL = E
			return False