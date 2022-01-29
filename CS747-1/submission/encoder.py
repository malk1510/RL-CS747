import sys

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

def encoder(states, policy, player):
	dic = {}
	for i in range(len(states)):
		dic[states[i]] = i
	dic['T'] = len(states)
	states.append('T')
	#print(ends('121221120'))
	print('numStates '+str(len(states)))
	print('numActions 9')
	print('end '+str(len(states)-1))
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
				print('transition '+str(dic[i])+' '+str(j)+' '+str(dic['T'])+' 0.0  1.0')
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
				print('transition '+str(dic[i])+' '+str(j)+' '+str(dic[new_st])+' '+term+' '+str(policy[curr_st][k]))
		print('mdtype episodic')
		print('discount 0.9')
	return


def main():
	inp = sys.argv
	policyFile = inp[2]
	stateFile = inp[4]
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
	encoder(states, policy, 3-player)

main()
