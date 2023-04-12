import pandas as pd
import random as rd


######################################## DATA ########################################################################################################################


def get_matrix(path):
        """Read the file and return the data in a matrix"""
        matrix = []
        with open (path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                matrix.append(line[:len(line)-1])
        return matrix

def get_list(map_matrix, char):
    """Return a list of the coordinates of the habitations"""
    habitation_list = []
    for i in range(len(map_matrix)):
        for j in range(len(map_matrix)):
            if map_matrix[i][j] == char:
                habitation_list.append((i,j))
    return habitation_list


######################################## GEN ########################################################################################################################


def generate_parent(free_fields,budget,cost_matrix,max_iter):
    parent = []
    parent_budget = 0
    while (parent_budget < budget):
        parent.append(rd.sample(free_fields-parent,1))
        cost = cost_matrix[parent[-1][0]][parent[-1][1]]
        if(parent_budget +cost > budget):
            parent.remove(parent[-1])
            for i in range(max_iter):
                field = rd.sample(free_fields-parent,1)
                cost = cost_matrix[field[0]][field[1]]
                if(parent_budget +cost <= budget):
                    parent_budget += cost
                    break
            break
        else:
            parent_budget += cost
    return parent


def generate_parents(free_fields,budget,cost_matrix,gen_length):
    parents = []
    for i in range(gen_length):
        parent = generate_parent(free_fields,budget,cost_matrix)
        parents.append(parent)
    return parents


def generate_child(parent1,parent2,free_fields,budget,cost_matrix):
    child = []
    parent1_budget = rd.sample(range(5,budget-5),1)
    parent2_budget = budget - parent1_budget
    return child

#def generate_new_gen(old_gen,score_list):



######################################## SCORE ########################################################################################################################


def average_distance(parent,habitation_list,distance_matrix):
    """Calculate the average distance between a field and the habitations"""
    total_distance = 0
    for field in parent:
        average_field_distance = 0 #average distance between one field and all the other habitations 
        if(distance_matrix[field[0]][field[1]] != -1):
            #if the distance between the field and the habitations has already been calculated
            average_field_distance = distance_matrix[field[0]][field[1]]
        else:
            total_field_distance = 0
            for j in range(len(habitation_list)):
                total_field_distance += abs(field[0] - habitation_list[j][0]) + abs(field[1] - habitation_list[j][1])
            average_field_distance = total_field_distance / len(habitation_list)
            distance_matrix[field[0]][field[1]] = average_field_distance   
        total_distance += average_field_distance
    average_distance = total_distance / len(parent)
    return average_distance


def production_score(parent,production_matrix):
    """Calculate the production score of a parent"""
    total_production = 0
    for i in range(len(parent)):
        total_production += production_matrix[parent[i][0]][parent[i][1]]
    return total_production


def compacity_score(parent):
    """Calculate the compacity score of a parent"""
    L = max(parent)[0] - min(parent)[0]
    l = max(parent,key=lambda x:x[1])[1] - min(parent,key=lambda x:x[1])[1]
    surface = L*l
    adjacent_score = 0
    for field in parent:
        if((field[0]-1,field[1]) in parent):
            adjacent_score += 1
        if((field[0]+1,field[1]) in parent):
            adjacent_score += 1
        if((field[0],field[1]-1) in parent):
            adjacent_score += 1
        if((field[0],field[1]+1) in parent):
            adjacent_score += 1
    return adjacent_score/surface


def calculate_score(parent,habitation_list,cost_matrix,production_matrix,distance_matrix):
    """Calculate the score of a parent"""
    #Init
    production_score = production_score(parent,production_matrix)
    habitation_average_distance = average_distance(parent,habitation_list,distance_matrix)
    compacity_score = compacity_score(parent)
    score = (production_score,habitation_average_distance,compacity_score)
    return score


######################################## MAIN ########################################################################################################################


def main():
    #Parameters
    rd.seed(123)
    gen_length = 100
    gen_nbr = 100
    budget = 50
    #Read the data
    cost_matrix = get_matrix('donnes_V2\Cost_map.txt')
    production_matrix = get_matrix('donnes_V2\Production_map.txt')
    map_matrix = get_matrix('donnes_V2\\Usage_map.txt')
    habitation_list = get_list(map_matrix,'C')
    free_fields = get_list(map_matrix,' ')
    distance_matrix = [[-1 for i in range(len(map_matrix))] for j in range(len(map_matrix))] # a matrix to store the average distance between a field and the habitations
    test = [(1,9),(3,4),(5,6)]
    print(compacity_score(test))
if __name__ == '__main__':
    main()