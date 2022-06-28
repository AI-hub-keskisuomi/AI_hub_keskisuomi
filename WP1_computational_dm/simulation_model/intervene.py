# -*- coding: utf-8 -*-
import magic
import population as pop
import scipy.stats as sta
import math
import random as ran
import os
import matplotlib.pyplot as pl

#class Purgatory:
	#people = []
	#def __init__(self):
		#self.people = []
	#def purge(self, person):
		#self.people.append(person)
	#def update(self):
		#for person in self.people:
			#p.monthlyreaper(person)
		#self.people = [person for person in self.people if person.dead != True]
#purgatory = False

class Followup:
	def __init__(self, customers):
		self.people = customers
		self.month = 1
	def update(self, population, money):
		for person in self.people:
			pop.monthlyreaper(person)
		self.people = [person for person in self.people if person.dead != True]
		if(self.month % magic.followupReassessmentPeriod == 0):
			if self.month != 12:
				money.spend(str(population.year) + " " + str(self.month) + " update of followup", len(self.people)*magic.followUpMonthlyCostPerPerson)
			for person in self.people:
				person.moneyspent = person.moneyspent + magic.followUpMonthlyCostPerPerson 
				if person.weight > 0 and sta.norm(1/27, math.sqrt(6)/27).ppf(ran.random())*10 < -0.05 and person.venegroup == 1: 
					person.weight -= 1
				elif person.weight > 0 and sta.norm(1/19, math.sqrt(6)/27).ppf(ran.random())*10 < -0.05 and person.venegroup == 2: 
					person.weight -= 1
				elif person.weight > 0 and sta.norm(1/17, math.sqrt(6)/27).ppf(ran.random())*10 < -0.05 and person.venegroup == 3: 
					person.weight -= 1
				if(pop.lottery(magic.followUpF2FAdditionalCounselingChance) and person.venegroup == 1):
					money.spend("Additional counseling at followup: " + str(person.hetu) + " in f2f", magic.followUpF2FAdditionalCounselingCost)
					person.moneyspent = person.moneyspent + magic.followUpF2FAdditionalCounselingCost
				elif(pop.lottery(magic.followUpDigiAdditionalCounselingChance) and person.venegroup == 2):
					money.spend("Additional counseling at followup: " + str(person.hetu) + " in digi", magic.followUpDigiAdditionalCounselingCost)
					person.moneyspent = person.moneyspent + magic.followUpDigiAdditionalCounselingCost
				elif(pop.lottery(magic.followUpAutoAdditionalCounselingChance) and person.venegroup == 3):
					money.spend("Additional counseling at followup: " + str(person.hetu) + " in auto", magic.followUpAutoAdditionalCounselingCost)
					person.moneyspent = person.moneyspent + magic.followUpAutoAdditionalCounselingCost
				if(self.month == 12):
					person.moneyspent = person.moneyspent + magic.followUpDoctorVisitCost
			if(self.month == 12):
				money.spend(str(population.year) + " " + str(self.month) + " checkup of followup", len(self.people)*magic.followUpDoctorVisitCost)		
		self.month = self.month + 1

class Diabetescare:
	def __init__(self, customers):
		self.people = customers
		self.month = 1
	def update(self, population, money):
		for person in self.people:
			pop.monthlyreaper(person)
		self.people = [person for person in self.people if person.dead != True]
		money.spend(str(population.year) + " " + str(self.month) + " update of diabetescare", len(self.people)*magic.diabetesCareCost)
		for person in self.people:
			person.moneyspent = person.moneyspent + magic.diabetesCareCost
		self.month = self.month + 1
		return
		        
class Autointervention:
	def __init__(self, customers, population):
		self.people = customers
		for person in self.people:
			person.venegroup = 3
			person.veneyear = population.year
		self.month = 1
	def update(self, population, money):
		for person in self.people:
			pop.monthlyreaper(person)
			if person.weight > 0 and sta.norm(-1/36, math.sqrt(3)/30).ppf(ran.random())*10 < -0.05:
				person.weight -= 1
		self.people = [i for i in self.people if i.dead != True]
		if(self.month == 1):
			money.spend(str(population.year) + " start of auto", len(self.people)*magic.autoStartingCost)
			for person in self.people:
				person.moneyspent = person.moneyspent + magic.autoStartingCost
		money.spend(str(population.year) + " " + str(self.month) + " update of auto", len(self.people)*magic.autoRunningCost)
		for person in self.people:
			person.moneyspent = person.moneyspent +  magic.autoRunningCost
		if(self.month % magic.interventionReassessmentPeriod == 0):
			for i in self.people:
				if(pop.lottery(magic.autoQuitChance)):
					self.people.remove(i)
					#purgatory.purge(i)
		self.month = self.month + 1
		return 
    
class Digiintervention:
	def __init__(self, customers, population):
		self.people = customers
		for person in self.people:
			person.venegroup = 2
			person.veneyear = population.year
		self.month = 1
	def update(self, population, money):
		for person in self.people:
			pop.monthlyreaper(person)
			if person.weight > 0 and sta.norm(-2/59, math.sqrt(3)/30).ppf(ran.random())*10 < -0.05:
					person.weight -= 1
		self.people = [person for person in self.people if person.dead != True]
		if self.month == 1:
			money.spend(str(population.year) + " start of digi", len(self.people)*magic.digiStartingCost)
			for person in self.people:
				person.moneyspent = person.moneyspent + magic.digiStartingCost
		money.spend(str(population.year) + " " + str(self.month) + " update of digi", len(self.people)*magic.digiRunningCost)
		for person in self.people:
			person.moneyspent = person.moneyspent + magic.digiRunningCost
		if(self.month % magic.interventionReassessmentPeriod == 0):
			for person in self.people:
				if(pop.lottery(magic.digiReadjustmentChance) and self.month != 12):                    
					money.spend("Readjustment of goals for: " + str(person.hetu) + " in digi", magic.digiReadjustmentCost)
					person.moneyspent = person.moneyspent + magic.digiReadjustmentCost
				if(pop.lottery(magic.digiQuitChance)):
					self.people.remove(person)
					#purgatory.purge(person)
				if(self.month == 12):
					if(pop.lottery(magic.digiEndCounselingChance)):
						money.spend("Additional counseling at end of f2f: " + str(person.hetu) + " in digi", magic.digiEndCounselingCost)
						person.moneyspent = person.moneyspent + magic.digiEndCounselingCost
		self.month = self.month + 1
		return
		
class F2fintervention:
	def __init__(self, customers, population):
		self.people = customers
		for person in self.people:
			person.venegroup = 1
			person.veneyear = population.year
		self.month = 1
	def update(self, population, money):
		for person in self.people:
			pop.monthlyreaper(person)
			if person.weight > 0 and sta.norm(-28/725, math.sqrt(3)/30).ppf(ran.random())*10 < -0.05:
				person.weight -= 1
		self.people = [person for person in self.people if person.dead != True]
		if(self.month == 1):
			money.spend(str(population.year) + " start of f2f", len(self.people)*magic.f2fStartingCost)
			for person in self.people:
				person.moneyspent = person.moneyspent + magic.f2fStartingCost
		money.spend(str(population.year) + " " + str(self.month) + " update of f2f", len(self.people)*magic.f2fRunningCost)
		for person in self.people:
			person.moneyspent = person.moneyspent + magic.f2fRunningCost
		if(self.month % magic.interventionReassessmentPeriod == 0):
			for person in self.people:
				if(pop.lottery(magic.f2fReadjustmentChance) and self.month != 12):                    
					money.spend("Readjustment of goals for: " + str(person.hetu) + " in f2f", magic.f2fReadjustmentCost)
					person.moneyspent = person.moneyspent + magic.f2fReadjustmentCost
				if(pop.lottery(magic.f2fQuitChance)):
					self.people.remove(person)
					#purgatory.purge(person)
				if(self.month == 12):
					if(pop.lottery(magic.f2fEndCounselingChance)):
						money.spend("Additional counseling at end of f2f: " + str(person.hetu) + " in f2f", magic.f2fEndCounselingCost)
						person.moneyspent = person.moneyspent + magic.f2fEndCounselingCost
		self.month = self.month + 1
		return
		
def runAYear(listOfProcesses, population, money):
	for month in range(1, 13):
		#purgatory.update()
		for process in listOfProcesses:
			process.update(population, money)

def sieve(population):
	intervened = []
	finalgroup = []
	for person in population:
		if(person.risc > magic.consideredHighRisk and person.gender == 1 and pop.lottery(magic.highRiskHasPrediabetesMen)):
			person.prediabetes = True
		elif(person.risc > magic.consideredHighRisk and person.gender == 2 and pop.lottery(magic.highRiskHasPrediabetesWomen)):
			person.prediabetes = True
		if person.prediabetes != False:
			finalgroup.append(person)			
		elif pop.lottery(100-magic.mediumRiskNoInterventionChance) and (person.risc < magic.consideredHighRisk and person.risc > magic.consideredRiskful):
			finalgroup.append(person)
		elif(pop.lottery(100-magic.highRiskNoInterventionChanceMen) and person.gender == 1 and person.risc >= magic.consideredHighRisk):
			finalgroup.append(person)
		elif(pop.lottery(100-magic.highRiskNoInterventionChanceWomen) and person.gender == 2 and person.risc >= magic.consideredHighRisk):
			finalgroup.append(person)
		for person in finalgroup:
			person.intervened = True
	return sorted(finalgroup, key = lambda person : person.risc, reverse = True)

def intervenedpopulation(population):
	intervened = sieve(population)
	f2f = [person for person in intervened if person.risc >= magic.consideredHighRisk or person.prediabetes != False]
	intervened = [person for person in intervened if person.risc < magic.consideredHighRisk]
	ratio = int(len(intervened)/magic.digiAutoInterventionRatio)
	digi = intervened[:ratio]
	auto = intervened[ratio:]
	return (f2f, digi, auto)

def intervene(population, money):
	#global purgatory
	#purgatory = Purgatory()
	alive = [person for person in population.people if person.dead != True]
	for person in population.people:
		person.intervened = False
	f2f, digi, auto = intervenedpopulation(alive)
	money.spend(str(population.year) + " sieve", magic.interventionSieveCost)
	care = [person for person in population.people if person.diag != False and person.dead != True]
	for person in care:
		person.intervened = True
	follow = []
	for person in alive:
		if person.diag != True and person.intervened != True and magic.interventionFollowUpYears >= (9999 if population.year == person.veneyear else population.year-person.veneyear):
			follow.append(person)
	for person in follow:
		person.intervened = True
	#for person in alive:
		#if person.intervened != True:
			#purgatory.purge(person)
	processList = [
	(F2fintervention(f2f, population), "f2f"),
	(Digiintervention(digi, population), "digi"),
	(Autointervention(auto, population), "auto"),
	(Diabetescare(care), "care"),
	(Followup(follow), "followup")
	]
	print(len([person for person in alive if person.intervened != False]))
	print(len(follow))
	runAYear([process[0] for process in processList if process[1] in magic.processesToRun], population, money)

	
class Graphics:
	def __init__(self):
		pass
		
	def line(self, data, title):
		pl.figure()
		pl.plot(data[0], data[1], data[2])
		pl.title(title)

	def histogram(self, data, title):
		pl.figure()
		pl.bar([*range(len(data[1]))], height=data[1])
		pl.xticks([*range(len(data[0]))], data[0])
		pl.title(title)
		
	def finalize(self):
		pl.show()

	def derive(self, data):
		b, m = npp.polyfit(data[0], data[1], 1)
		pl.plot(np.array(data[0]), b + np.array(data[1]) * m, "-")

class Money:
	f2fmoney = 0
	f2fledger = {}
	digimoney = 0
	digiledger = {}
	automoney = 0
	autoledger = {}
	followmoney = 0 
	followledger = {}
	caremoney = 0
	careledger = {}
	othermoney = 0
	otherledger = {}
	
	def __init__(self):
		self.f2fmoney = 0
		self.digimoney = 0
		self.automoney = 0
		self.followmoney = 0
		self.caremoney = 0
		self.othermoney = 0
		
	def spend(self, reason, amount):
		if " followup" in reason:
			self.followmoney = self.followmoney + amount
			self.followledger.update({reason : amount})
		else:
			if " f2f" in reason:
				self.f2fmoney = self.f2fmoney + amount
				self.f2fledger.update({reason : amount})
			elif " digi" in reason:
				self.digimoney = self.digimoney + amount
				self.digiledger.update({reason : amount})
			elif " auto" in reason:
				self.automoney = self.automoney + amount
				self.autoledger.update({reason : amount})
			elif " diabetescare" in reason:
				self.caremoney = self.caremoney + amount
				self.careledger.update({reason : amount})
			else:
				self.othermoney = self.othermoney + amount
				self.otherledger.update({reason : amount})
	def total(self):
		return sum(self.followledger.values())+sum(self.f2fledger.values())+ \
				sum(self.digiledger.values())+ sum(self.autoledger.values())+ \
				sum(self.careledger.values())+sum(self.otherledger.values())
				
	def report(self, filename):
		filenumber = 0
		while(os.path.isfile(magic.simulationSaveFileName + str(filenumber) + ".txt")):
			filenumber = filenumber + 1
		with open('simuldata/' + magic.simulationSaveFileName + str(filenumber) + '.txt', 'w') as f:
			print("In total: " + str(self.total()), end="\n", file=f)
			print("Of which in:", end="\n", file=f)
			print("F2F: " + str(self.f2fmoney), end="\n", file=f)
			print("Digi: " + str(self.digimoney), end="\n", file=f)
			print("Auto: " + str(self.automoney), end="\n", file=f)
			print("Followup: " + str(self.followmoney), end="\n", file=f)
			print("Diabetes Care: " + str(self.caremoney), end="\n", file=f)
			print("Other costs: " + str(self.othermoney), end="\n", file=f)
			print(self.f2fledger, file=f)
			print(self.digiledger, file=f)
			print(self.autoledger, file=f)
			print(self.followledger, file=f)
			print(self.careledger, file=f)
			print(self.otherledger, file=f)
			
	def listify(self):
		return (["care", "f2f", "digi", "auto", "followup", "other"],
				[self.caremoney, self.f2fmoney, self.digimoney, self.automoney, self.followmoney, self.othermoney])
