import pickle
import matplotlib.pyplot as plt
import numpy as np
import os


container = []
for i in range(100):
    if os.path.exists(f'./results/final/skip-neg/accuracy_per_category_{i}'):
        temp = pickle.load(open(f'./results/final/skip-neg/accuracy_per_category_{i}', 'rb'))
        container.append(temp)


capital_common_countries = []
gram4_superlative = []
family = []
capital_world = []
gram4 = []

for i in range(len(container)):
    capital_common_countries.append(container[i]['capital-common-countries'])
    gram4_superlative.append(container[i]['gram4-superlative'])
    family.append(container[i]['family'])
    capital_world.append(container[i]['capital-world'])
    gram4.append(container[i]['gram4-superlative'])
    

x = [i for i in range(len(container))]

plt.plot(x, capital_common_countries, label = 'capital-common-countries')
plt.plot(x, gram4_superlative, label='gram4-superlative')
plt.plot(x, family, label='family')
plt.plot(x, capital_world, label='capital-world')
plt.plot(x, gram4, label='gram4-superlative')

plt.xlabel('% trained')
plt.ylabel('accuracy')

plt.title('Monitoring accuracy')
plt.legend()

plt.savefig('./results/testing.png')