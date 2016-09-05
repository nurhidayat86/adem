import subprocess;

command = "python pcasvmcvconf.py ";
tr = [0.6, 0.7];
sr = [1, 5, 10, 30, 60, 300, 600, 900, 1800];
fl = [300, 600, 900, 1800, 3600];
h = ['r2'];

for house in h:
	for test in tr:
		for sampling in sr:
			for feature in fl:
				cmd = command + "--house=" + house + " --tr=" + str(test) + " --sr=" + str(sampling) + " --fl=" + str(feature);
				print(cmd);
				subprocess.call(cmd, shell=True);