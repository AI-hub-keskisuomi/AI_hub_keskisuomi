import scipy.stats as sta
import random as ran
import math

def test_func(mu, std):
	rep = 1000
	variables = []
	over = []
	for i in range(0, rep):
		var = sta.norm(mu,std).ppf(ran.random())*10 
		variables.append(var)
		if var < -0.05:
			over.append(var)
	return (len(over) / len(variables))

def main(): 
	#F2F - Weight loss
	mu = -28/725
	std = math.sqrt(3)/30
	print ("F2F weight loss: " + str(test_func(mu, std)))
	#Digi - Weight loss
	mu = -2/59
	std = math.sqrt(3)/30
	print ("Digi weight loss: " + str(test_func(mu, std)))
	#Auto - Weight loss
	mu = -1/36
	std = math.sqrt(3)/30
	print ("Auto weight loss: " + str(test_func(mu, std)))

if __name__== "__main__":
	main()  


