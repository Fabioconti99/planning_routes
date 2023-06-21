import numpy as np
import matplotlib.pyplot as plt
from docplex.mp.model import Model
import docplex.mp.solution as Sol

#active_arcs = []
def get_visited_nodes(nodes_list):
    # Sort the nodes list based on the first element of each tuple
    nodes = np.array(nodes_list)
    perm = np.zeros([1,nodes.shape[0]])

    perm[0,1]=nodes[0,1]

    for i in range(nodes.shape[0]-2): 

        perm[0,i+2] = nodes[int(perm[0,i+1]),1]
        
    return perm


# normal tsp function

def tsp_cplex_mod(points, arr_constraints, time_limit):

    #global active_arcs

    n = len(points)

    cities = [i for i in range(n)]

    arcs = [(i,j) for i in cities for j in cities if i!=j]

    # creation of the data 
    coord_x = points[:, 0]  # First column
    coord_y = points[:, 1]  # Second column

    constrained_cities = []

    for i in range(len(arr_constraints)):

        constrained_cities.append(tuple(arr_constraints[i,:]))


    # printing the points 

    dist = {(i,j):np.hypot(coord_x[i]-coord_x[j],coord_y[i]-coord_y[j]) for i,j in arcs}


    # Find the maximum distance in the original distance matrix
    max_dist = max(dist.values())

    # Create a new distance matrix that adds the maximum distance to all the pairs of nodes that are not constrained
    new_dist = {(i,j): dist[(i,j)] if (i,j) in constrained_cities or (j,i) in constrained_cities else dist[(i,j)] + max_dist for i,j in arcs}


    # C-plex Model creation:

    mdl=Model('TSP')

    # Creation of the cplex decision variable: 

    x = mdl.binary_var_dict(arcs,name='x')
    d = mdl.continuous_var_dict(cities,name='d')

    # Set the time limit

    mdl.parameters.timelimit = time_limit


    # Creation of the objective function

    mdl.minimize(mdl.sum(new_dist[i]*x[i] for i in arcs))

    # Constraints

    # out
    for c in cities: 
        mdl.add_constraint(mdl.sum(x[(i,j)] for i,j in arcs if i == c)==1, ctname='out_%d'%c)

    # in
    for c in cities: 
        mdl.add_constraint(mdl.sum(x[(i,j)] for i,j in arcs if j == c)==1, ctname='in_%d'%c)


    # big M but better
    for i,j in arcs:

        if j!=0: 
            mdl.add_indicator(x[(i,j)],d[i]+1==d[j],name = 'order_(%d,_%d)' %(i,j))

    for i,j in constrained_cities:
        for k in cities:
            if k != i and k != j:
                mdl.add_indicator(x[(i,k)], x[(i,j)] <= x[(j,k)], name='constrained_(%d,%d)_(%d)' % (i,j,k))

    # passing the complete model 

    mdl.export_to_string()

    # solving the model

    sol_1= mdl.solve(log_output=True)

    # is this the optimal solution? 

    mdl.get_solve_status()

    # which is the final solution?

    sol_1.display()

    active_arcs = [i for i in arcs if x[i].solution_value>0.9]

    active_arcs = get_visited_nodes(active_arcs)
    
    return active_arcs.astype(int).flatten().tolist()


# normal tsp function

def tsp_cplex(points, time_limit):


    #global active_arcs


    n = len(points)

    cities = [i for i in range(n)]

    arcs = [(i,j) for i in cities for j in cities if i!=j]

    # creation of the data 
    coord_x = points[:, 0]  # First column
    coord_y = points[:, 1]  # Second column

    # printing the points 

    dist = {(i,j):np.hypot(coord_x[i]-coord_x[j],coord_y[i]-coord_y[j]) for i,j in arcs}


    # C-plex Model creation:

    mdl=Model('TSP')

    # Creation of the cplex decision variable: 

    x = mdl.binary_var_dict(arcs,name='x')
    d = mdl.continuous_var_dict(cities,name='d')

    mdl.parameters.timelimit = time_limit


    # Creation of the objective function

    mdl.minimize(mdl.sum(dist[i]*x[i] for i in arcs))

    # Constraints

    # out
    for c in cities: 
        mdl.add_constraint(mdl.sum(x[(i,j)] for i,j in arcs if i == c)==1, ctname='out_%d'%c)

    # in
    for c in cities: 
        mdl.add_constraint(mdl.sum(x[(i,j)] for i,j in arcs if j == c)==1, ctname='in_%d'%c)


    # big M but better
    for i,j in arcs:

        if j!=0: 
            mdl.add_indicator(x[(i,j)],d[i]+1==d[j],name = 'order_(%d,_%d)' %(i,j))


    # passing the complete model 

    mdl.export_to_string()

    # solving the model

    sol_1= mdl.solve(log_output=True)

    # is this the optimal solution? 

    mdl.get_solve_status()

    # which is the final solution?

    sol_1.display()

    active_arcs = [i for i in arcs if x[i].solution_value>0.9]

    active_arcs = get_visited_nodes(active_arcs)
    
    return active_arcs.astype(int).flatten().tolist()
    

'''
coord_x = np.array([1085.46688765,777.45044273, 1622.62914711 , 543.35707197, 1376.65586748, 634.3305711, 1479.64086445 , 421.38107218, 1257.64175497, 481.35269641  , 1326.66964245  ,595.3712029, 1225.64515452, 386.37678324, 1229.63713277, 716.35498525, 1561.6735792  , 911.37993181, 1221.64350569 , 826.38903294 , 1673.62756862 , 610.47112469 , 1457.65973468 ,500.36314568, 1349.66810432, 405.38104814 , 1253.62639815  , 651.36977676 , 1498.64633652 , 811.3783469, 1649.62507107 , 922.38831895 , 1637.6180287 ,668.35264779 ,1525.64759635 ,354.35449457 ,1213.67553473  ,692.47928839 ,1549.64937372 ,524.33572738  ,1375.68954117  , 757.342783   ,1611.64330098 , 726.38861075 ,1582.63433847 , 821.35262257 ,1662.63004345 , 457.4283123, 1310.65933418  , 810.35468387, 1644.64604855, 572.42733947, 876.57266053, 452.32665121, 1290.67530191, 1119.42733947, 1421.57266053])
coord_y = np.array([1150.12382864,1749.18379121,1754.71977325,1045.3763871,1053.269609,1267.00611415,1272.54295811,632.82322754,638.30079589,840.34361678,845.88050431,359.66172098,363.79006125,494.98591129,500.50932797,1541.54881158,1547.08570991,293.78953957,295.82178856,2024.35856374,2029.90803782,1197.53340986,1203.08255693,908.21861742,913.78162672,564.40305691,569.9591257 ,1336.8514812 ,1342.40120435,1887.90168412,1893.39226119,2094.09338768,2098.77819435,1404.03701755,1411.9644473,427.45571185,433.08432722,1473.76158443,1479.37611089,977.64312658,985.69647303,1680.56663666,1686.16236724,1611.50069572,1617.10916756,1818.30106024,1823.81148859,770.82520522,776.41393052,1955.2179828,1960.68265197,1115, 1115,  703.13444357,  708.62568826, 1120, 1120])

points = np.zeros([57,2])
points[:,0]=coord_x
points[:,1]=coord_y

constrained_cities = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],[31, 32],[33, 34],[35, 36],[37, 38],[39, 40],[41, 42],[43, 44],[45, 46],[47, 48],[49, 50],[51, 52],[53, 54],[55, 56]])


print(tsp_cplex(points,10))
#print(tsp_cplex_mod(points, constrained_cities,10))

#print(tsp_cplex(points,10))


plt.figure()

plt.xlabel("X coord")
plt.ylabel("Y coord")
plt.title("TSP solition")

plt.scatter(x=coord_x,y=coord_y, color='blue', zorder=1)

for i,j in active_arcs:
    plt.plot([coord_x[i],coord_x[j]], [coord_y[i],coord_y[j]], color='blue',zorder=1)

plt.show()
'''