# -*- coding: utf-8 -*-
import magic
import random as ran
import population as pop
import intervene as inter

def controller():
	population = pop.Population(magic.simulationStartYear) 
	print("Luotu Suomen väestö " + str(population.year))
	money = inter.Money()
	gfx = inter.Graphics()
	totals = []
	lasttotal = 0
	for year in range(magic.yearsPerIterationOfSimulation):	
		inter.intervene(population, money)
		print("Interventoitu " + str(population.year))
		population.update()
		total = money.total()
		delta = total-lasttotal
		totals.append(delta)
		lasttotal = total
	print("Deltas: " + str(totals))
	money.report(magic.simulationSaveFileName)
	gfx.line(([*range(magic.simulationStartYear, magic.simulationStartYear+magic.yearsPerIterationOfSimulation)], totals, "-"), "Kustannusten kasvu")
	gfx.histogram(money.listify(), "Kustannusten jakautuminen")
	gfx.histogram((["naiset", "miehet"], 
	[sum(pop.femaledeathages)/float(len(pop.femaledeathages)), sum(pop.maledeathages)/float(len(pop.maledeathages))]),
	"Keskielinikä")
	women = [woman.moneyspent for woman in population.people if woman.gender == 2]
	men = [man.moneyspent for man in population.people if man.gender == 1]
	gfx.histogram((["naiset", "miehet"], 
	[sum(women)/float(len(women)), sum(men)/float(len(men))]),
	"Rahan keskimääräinen käyttö per henkilö")
	gfx.finalize()

def main():
	ran.seed(1) 
	controller()
	
if __name__== "__main__":
	main()  

