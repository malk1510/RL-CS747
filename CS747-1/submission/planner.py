import pulp
import numpy as np
import sys
def iteration(s,a,t,r,gamma):
	n = len(s)
	n_act = len(a)
	v_prev = [0.0 for i in s]
	v_new = [0.0 for i in s]
	pi = [0 for i in s]
	for i in range(n):
		for j in range(n_act):
			sm = 0
			for k in range(n):
				sm += t[i][j][k]*(r[i][j][k] + gamma*v_prev[k])
			if((sm>v_new[i]) or (j==0)):
				v_new[i] = sm
				pi[i] = j
	tme=1
	while(v_new != v_prev):
		v_prev = [i for i in v_new]
		v_new = [0.0 for i in s]
		for i in range(n):
			for j in range(n_act):
				sm = 0
				for k in range(n):
					sm += t[i][j][k]*(r[i][j][k] + gamma*v_prev[k])
				if((sm>v_new[i]) or (j==0)):
					v_new[i] = sm
					pi[i] = j
		tme+=1
	return {'Value':v_new, 'Policy':pi}

def lin_prog(s,a,t,r,gamma):
	n = len(s)
	n_act = len(a)
	lpp = pulp.LpProblem('MDP-LP', pulp.LpMinimize)
	v = []
	for i in range(n):
		v.append(pulp.LpVariable(('v'+str(i)), lowBound=0, cat='Continuous'))
	for i in range(n):
		for j in range(n_act):
			x = 0
			for k in range(n):
				x += t[i][j][k]*(r[i][j][k] + gamma*v[k])
			lpp += (v[i] >= x), ('C'+str(i)+'_'+str(j))
	lpp.solve()
	v_ans = []
	pi = [0 for i in s]
	for i in range(1,n+1):
		v_ans.append(lpp.variables()[i].varValue)
	for i in range(n):
		max_sm = 0
		for j in range(n_act):
			sm = 0
			for k in range(n):
				sm += (t[i][j][k])*(r[i][j][k] + gamma*v_ans[k])
			if((sm>max_sm) or (j==0)):
				pi[i] = j
				max_sm = sm
	return {'Value':v_ans, 'Policy':pi}

def hpi(s,a,t,r,gamma):
	imp_acts = [[] for i in s]
	imp_states = []
	n = len(s)
	n_act = len(a)
	pi = [0 for i in s]
	coeff = np.array([[0.0 for i in s] for j in s])
	y = np.array([0.0 for i in s])
	for i in range(n):
		sm = 0.0
		for j in range(n):
			coeff[i][j] = -gamma*t[i][pi[i]][j]
			if(i==j):
				coeff[i][j]+=1
			sm += t[i][pi[i]][j]*r[i][pi[i]][j]
		y[i] = sm
	v = list(np.linalg.solve(coeff, y))
	#print(v)

	#for i in range(n):
	#	q = 0
	#	for j in range(n):
	#		q += t[i][pi[i]][j]*(r[i][pi[i]][j] + gamma*v[j])
	#	q -= v[i]
	#	print(q)

	for i in range(n):
		for j in range(n_act):
			q = 0
			for k in range(n):
				q += t[i][j][k]*(r[i][j][k] + gamma*v[k])
			if (q>(v[i]+10**-10)):
				imp_acts[i].append(j)
	#		print(q,end=' ')
	#	print()
		if(len(imp_acts[i])>0):
			imp_states.append(i)
	#print(imp_states)
	#print(imp_acts)
	while(len(imp_states)>0):
	#	print(pi)
		pi[imp_states[0]] = imp_acts[imp_states[0]][0]
		imp_acts = [[] for i in s]
		imp_states = []
		coeff = np.array([[0.0 for i in s] for j in s])
		y = np.array([0.0 for i in s])
		for i in range(n):
			sm = 0.0
			for j in range(n):
				coeff[i][j] = -gamma*t[i][pi[i]][j]
				if(i==j):
					coeff[i][j]+=1
				sm += t[i][pi[i]][j]*r[i][pi[i]][j]
			y[i] = sm
		v = list(np.linalg.solve(coeff, y))
		for i in range(n):
			for j in range(n_act):
				q = 0.0
				for k in range(n):
					q += t[i][j][k]*(r[i][j][k] + gamma*v[k])
				if (q>(v[i]+10**-10)):
					imp_acts[i].append(j)
			if(len(imp_acts[i])>0):
				imp_states.append(i)
	return {'Value':v, 'Policy':pi}


############################################################

def task1(file, algo):
	with open(file, 'r') as f:
		s = range(int(f.readline().split()[1]))
		a = range(int(f.readline().split()[1]))
		end = f.readline()
		t = [[[0.0 for i in s] for j in a] for k in s]
		r = [[[0.0 for i in s] for j in a] for k in s]
		while True:
			trans_str = f.readline().split()
			if(trans_str[0] != 'transition'):
				break
			i = int(trans_str[1])
			j = int(trans_str[2])
			k = int(trans_str[3])
			t[i][j][k] = float(trans_str[5])
			r[i][j][k] = float(trans_str[4])
		gamma = float(f.readline().split()[1])
	mp = {'vi':iteration, 'lp':lin_prog, 'hpi':hpi}
	dic = mp[algo](s,a,t,r,gamma)
	for i in s:
		print(str(dic['Value'][i]) + " " + str(dic['Policy'][i]))
	return


def main():
	arguments = sys.argv
	addr = arguments[2]
	if len(arguments)>3:
		algo = arguments[4]
	else:
		algo = 'hpi'
	task1(addr, algo)
	return

main()