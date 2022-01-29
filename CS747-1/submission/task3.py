import sys
import numpy as np
def ends(st):
	x1 = float(reward(st,1))
	x2 = float(reward(st,2))
	if (x1==1.0) or (x2==1.0):
		return True
	st = list(st)
	if not '0' in st:
		return True
	return False
def reward(st, player):
	if(st=='T'):
		return '1.0'
	p = str(3-player)
	s = list(st)
	rw = 0.0
	b3 = True
	b4 = True
	for i in range(3):
		b1 = True
		b2 = True
		b3 = b3 and (s[2*i+2]==p)
		b4 = b4 and (s[4*i]==p)
		for j in range(3):
			b1 = b1 and (s[3*i+j]==p)
			b2 = b2 and (s[i+3*j]==p)
		if(b1 or b2):
			rw = 1.0
	if(b3 or b4):
		rw = 1.0
	return str(rw)

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

def task1_file(file, algo, ans_file):
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
	mp = {'hpi':hpi}
	dic = mp[algo](s,a,t,r,gamma)
	with open(ans_file,'a') as f:
		for i in s:
			f.write(str(dic['Value'][i]) + " " + str(dic['Policy'][i])+'\n')
	return

def encoder_tofile(states, policy, player, new_policy_file):
	dic = {}
	for i in range(len(states)):
		dic[states[i]] = i
	dic['T'] = len(states)
	states.append('T')
	#print(ends('121221120'))
	with open(new_policy_file,'a') as f:
		f.write('numStates '+str(len(states))+'\n')
		f.write('numActions 9\n')
		f.write('end '+str(len(states)-1)+'\n')
		for i in states:
			if(i=='T'):
				continue
			for j in range(9):
				curr_st = list(i)
				if(curr_st[j]!='0'):
					continue
				curr_st[j] = str(player)
				curr_st = ''.join(curr_st)
				if(ends(curr_st)):
					f.write('transition '+str(dic[i])+' '+str(j)+' '+str(dic['T'])+' 0.0  1.0\n')
					continue
				for k in range(9):
					if(policy[curr_st][k]==0.0):
						continue
					new_st = [l for l in list(curr_st)]
					new_st[k] = str(3-player)
					new_st = ''.join(new_st)
					term = reward(new_st, player)
					if(ends(new_st)):
						new_st = 'T'
					f.write('transition '+str(dic[i])+' '+str(j)+' '+str(dic[new_st])+' '+term+' '+str(policy[curr_st][k])+'\n')
		f.write('mdtype episodic\n')
		f.write('discount 0.9\n')
	return

def decode_file(ans_file, statesFile, policyFile, player):
	states = []
	actions = []
	with open(statesFile, 'r') as f:
		for line in f:
			states.append(line)
	with open(ans_file, 'r') as f:
		for _ in states:
			[value, pi] = f.readline().split()
			actions.append(int(pi))
	with open(policyFile, 'a') as f:
		f.write(str(player)+"\n")
		for i in range(len(states)):
			x = [0 for i in range(9)]
			x[actions[i]] = 1
			x = ' '.join(x)
			f.write(states[i]+' '+x+'\n')
	return

def main(stateFile, policyFile, new_policy_file):
	policy = {}
	states = []
	with open(policyFile,'r') as f:
		player = int(f.readline())
		while True:
			try:
				x = f.readline()[:-1].split()
				policy[x[0]] = [float(i) for i in x[1:]]
			except:
				break
	with open(stateFile,'r') as f:
		for line in f:
			states.append(line[:-1])
	encoder_tofile(states, policy, 3-player, new_policy_file)
	return

def task3(state_files, policy_init, player_init):
	states = state_files[player_init-1]
	policies = policy_init
	player = player_init
	for i in range(10):
		main(states, policies, ('new_policy'+str(i)+'.txt'))
		task1_file(('new_policy'+str(i)+'.txt'), 'hpi', ('policy_values'+str(i)+'.txt'))
		policies = 't2_policy'+str(i)+'.txt'
		decode_file( ('policy_values'+str(i)+'.txt'), states, policies, player)
		player = 3-player
		states = state_files[player]
	return

state_files = ['data/attt/states/states_file_p1.txt', 'data/attt/states/states_file_p2.txt']
policy_init = 'data/attt/policies/p1_policy1.txt'
task3(state_files, policy_init, 2)
