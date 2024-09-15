import numpy as np
import pandas as pd 

# disk magnets
# diameter, thickness, number, price (CHF)
d = [[1,.5,100,8.49],
     [1,.5,100,4.43], # https://de.aliexpress.com/item/1005004052561992.html?mp=1&gatewayAdapt=glo2deu
     [1,.5,2000,53.19],
     [4,2,50,1.21],
     [10,1,20,1.55],
     [6.5,1.58,100,3.68],
     [5,2,200,8.26],
     [10,1,20, 1.37],
     [10,1,50, 2.37],
     [10,1,100, 3.72],
     [10,2,20,1.95],
     [10,2,50,3.48],
     [10,2,100,6.68],
     [10,3,20,2.45],
     [10,3,50,5.20],
     [10,3,100,10.77],
     [8,1,2,.63],
     [8,1,100,2.39],
     [8,5,100,9.02],
     [8,6,100,9.89],
     [8,8,100,16.09],
     [8,10,100,16.09],
     [8,15,100,22.19],
     [8,20,100,27.99],
     [5,1,150,6.69],
     [6,2,150,10.39],
     [12,1.5,2000,124.99],
     [12,2,500,39.79],
     [10,1.5,500,20.99],
     [4,2,1000,10.03],
     [8,2,1000,33.99],
     [12,2,1000,57.99], # this is the best without VAT https://de.aliexpress.com/item/1005006362930902.html?algo_exp_id=f59f0c06-4fd7-45f2-9553-930639b92215-9&utparam-url=scene%3Asearch%7Cquery_from%3A
     [15,2,500,45.59],
     [20,2,500,68.39]
     ]


# block magnets
# width, height, thickness, number, price (CHF)
b = [[20, 10, 3, 10, 1.55],
     [20, 10, 3, 20, 2.25]];

df = pd.DataFrame(d, columns=['diameter', 'thickness', 'number', 'price'])
df['volume'] = np.pi * df['diameter']**2 * df['thickness'] * df['number']
df['price_per_cm3']=  1000 * df['price'] / df['volume']

# sort by price per mm3

dfs= df.sort_values(by='price_per_cm3', ascending=True)
print(dfs)

bf = pd.DataFrame(b, columns=['width', 'height', 'thickness', 'number', 'price'])
bf['volume'] = bf['width'] * bf['height'] * bf['thickness'] * bf['number']
bf['price_per_cm3']=  1000 * bf['price'] / bf['volume']
bfs = bf.sort_values(by='price_per_cm3', ascending=True)
print(bfs)
