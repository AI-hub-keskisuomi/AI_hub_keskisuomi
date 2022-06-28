# -*- coding: utf-8 -*-
import pandas
import magic
import random as ran

def readfile(file_):
	return pandas.read_csv('csv/'+file_, delimiter=";", low_memory=False)

#HUOM!!!! - NIMET MUOKATTU
vaestonkasvu = readfile("vaestonkasvu.csv")
ikapolaatio = readfile("ikapolaatio.csv")
miesyli = readfile("miesyli.csv")
naisyli = readfile("naisyli.csv")
miesliha = readfile("miesliha.csv")
naisliha = readfile("naisliha.csv")
miesvyotaro = readfile("miesvyotaro.csv")
naisvyotaro = readfile("naisvyotaro.csv")
mieslii = readfile("mieslii.csv")
naislii = readfile("naislii.csv")
miesruoka = readfile("miesruoka.csv")
naisruoka = readfile("naisruoka.csv")
miesveri = readfile("miesveri.csv")
naisveri = readfile("naisveri.csv")
diabetesriski = readfile("diabetesriski.csv")
kuolema = readfile("kuolema.csv")
riskpolaatio = readfile("riskpolaatio.csv")

maledeathages = []
femaledeathages = []


def stork(population):
	hetu = 0
	for person in population.people:
		if person.hetu > hetu:
			hetu = person.hetu	
	maleages, femaleages = agespread(population.year)
	for boy in range(0, int(maleages[0]*magic.populationMultiplier)):
		hetu = hetu + 1
		population.people.append(Person(1, 0, hetu))
	for girl in range(0, int(femaleages[0]*magic.populationMultiplier)):
		hetu = hetu + 1
		population.people.append(Person(2, 0, hetu))

def diagnose(person):
    if lottery(100*(1-((1-(riskpolaatio.iloc[int(min(21, person.risc)), 1]))**(1.0/10.0)))):
        return True
    return False
    
def monthlyreaper(person):        
	if lottery(100*(1-((1-(((kuolema.iloc[int(min(person.age, 99)), person.gender]/10)*person.deathmultiplier)/100))**(1.0/12.0)))):
		person.dead = True
	if person.dead != False and person.gender == 1:
		maledeathages.append(person.age)
	elif person.dead != False and person.gender == 2:
		femaledeathages.append(person.age)

def getdiabetesatstart(person):
	return lottery(diabetesriski.iloc[max(min(person.age, 64), 7)-7, 0])

#HUOM!!!! - MUOKATTU
def findrisc(person):
	return sum([person.ageri, person.weight, person.waistline, person.exercise,
					person.food, person.hbp, person.hbs, person.ancestry])

def lottery(chance):
    return (ran.randint(0, 1000000000)/10000000.0) < chance

def finterveysriski(person, data):
	ika = person.age
	if person.age > 29 and person.age < 34:
		ika = 34
	elif person.age <= 29:
		return 0.0
	else:
		return lottery(data.iloc[min(80, ika)-34, 1])

#HUOM!!!! - MUOKATTU
def ancestryrisk(person):
	for parent in range(0, 3):
		if lottery(magic.diabetesPrevalance):
			return 5.0
	for uncles_and_aunts in range(0, 2):
		if lottery(magic.diabetesPrevalance):
			return 5.0
	if person.age > 25:
		if person.age > 64:
			age = 64
		else:
			age = person.age
		for children in range(0, 3):
			if lottery(diabetesriski.iloc[diabetesriski.iloc[:,1].tolist().index(age-18), 0]):
				return 5.0
	for grandparent in range(0, 5):
		if lottery(magic.diabetesPrevalance):
			return 3.0
	for cousins in range(0, 3):
		if lottery(magic.diabetesPrevalance):
			return 3.0
	return 0.0  

#HUOM!!!! - MUOKATTU	
def hbsrisk(person):
	#Raskausdiabetes
	#https://www.terveyskirjasto.fi/terveyskirjasto/tk.koti?p_artikkeli=dlk00168
	if person.gender == 2:
		if person.prediabetes != False: 
			return 5.0
		elif lottery(magic.highBloodSugarRiskWomen):
			return 5.0
	return 0.0

#FINTEREVEYS
def hbprisk(person):
	if person.gender == 1:
		if finterveysriski(person, miesveri):
			return 2.0
	else:
		if finterveysriski(person, naisveri):
			return 2.0
	return 0.0
	
#FINTEREVEYS
def foodrisk(person): 
	if person.gender == 1:
		if finterveysriski(person, miesruoka):
			return 0.0
	else:
		if finterveysriski(person, naisruoka):
			return 0.0
	return 1.0

#FINTEREVEYS
def exerciserisk(person):
	global maleexe, femaleexe
	if person.gender == 1:
		if finterveysriski(person, mieslii):
			return 1.0
	else:
		if finterveysriski(person, naislii):
			return 1.0
	return 0.0
				
#FINTEREVEYS
def waistlinerisk(person):
	if person.gender == 1:
		if finterveysriski(person, miesvyotaro):
			if lottery(magic.waistlineLargeRiskMen):
				return 4.0
			else:
				return 3.0
	else:
		if finterveysriski(person, naisvyotaro):
			if lottery(magic.waistlineLargeRiskWomen):
				return 4.0
			else:
				return 3.0
	return 0.0

#FINTEREVEYS
def weightrisk(person):
	if person.age < 34:
		return 0.0
	elif person.gender == 1:
		if lottery(miesyli.iloc[int(min(person.age, 80))-34, 1]):
			if lottery(miesliha.iloc[int(min(person.age, 80))-34, 1]):
				return 3.0
			else:
				return 1.0
	else:
		if lottery(naisyli.iloc[int(min(person.age, 80))-34, 1]):
			if lottery(naisliha.iloc[int(min(person.age, 80))-34, 1]):
				return 3.0
			else:
				return 1.0
	return 0.0

#FINTEREVEYS
#HUOM!!!! - LISÄTTY
def agerisk(person):
	if person.age < 45: 
		return 0.0
	return ikapolaatio.iloc[min(65, person.age)-45, 1]

class Person:
	risc = None
	diag = False
	prediabetes = False
	dead = False
	venegroup = None
	veneyear = 0
	reaped = False
	deathmultiplier = 1.0
	moneyspent = 0.0	
	intervened = False
	
	def __init__(self, gender, age, hetu):
		self.gender = gender
		self.age = age
		self.hetu = hetu
		self.ageri = agerisk(self)
		self.weight = weightrisk(self)
		self.waistline = waistlinerisk(self)
		self.exercise = exerciserisk(self)
		self.food = foodrisk(self)
		self.hbp = hbprisk(self)
		self.hbs = hbsrisk(self)
		self.ancestry = ancestryrisk(self)
		self.risc = findrisc(self)
		self.diag = getdiabetesatstart(self)
		if self.diag != False:
			if self.gender == 1:
				self.deathmultiplier = magic.diabetesDeathMultiplierMen
			else:
				self.deathmultiplier = magic.diabetesDeathMultiplierWomen
				
def agespread(year):
	yearindex =  vaestonkasvu["Vuosi"].tolist().index(year)
	maleages = vaestonkasvu.iloc[yearindex, 1:102].tolist()
	femaleages = vaestonkasvu.iloc[yearindex, 102:203].tolist()
	return (maleages, femaleages)

class Population:
	year = 0
	people = []
	
	def __init__(self, year):
		self.year = year
		maleages, femaleages = agespread(self.year)
		hetu = 1
		for age in range(1, 101):
			for amount in range(0, int(maleages[age]*magic.populationMultiplier)):
				hetu = hetu + 1
				self.people.append(Person(1, age, hetu))
			for amount in range(0, int(femaleages[age]*magic.populationMultiplier)): 
				hetu = hetu + 1
				self.people.append(Person(2, age, hetu))	
		
	def update(self):
		self.year = self.year + 1
		for person in self.people:
			person.age = person.age + 1
			if(person.diag != True):
				person.diag = diagnose(person)
			if(person.diag != False):
				if person.gender == 1:
					person.deathmultiplier = magic.diabetesDeathMultiplierMen
				else:
					person.deathmultiplier = magic.diabetesDeathMultiplierWomen
		stork(self)
