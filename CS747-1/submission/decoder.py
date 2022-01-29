import sys

def decode(ans_file, statesFile, player):
	states = []
	actions = []
	with open(statesFile, 'r') as f:
		for line in f:
			states.append(line)
	with open(ans_file, 'r') as f:
		for _ in states:
			[value, pi] = f.readline().split()
			actions.append(int(pi))

	print(str(player))
	for i in range(len(states)):
		x = [0 for i in range(9)]
		x[actions[i]] = 1
		x = ' '.join([float(i) for j in x])
		print(states[i]+' '+x)
	return


args = sys.argv
ans_file = args[2]
statesFile = args[4]
player = int(args[6])
decode(ans_file, statesFile, player)