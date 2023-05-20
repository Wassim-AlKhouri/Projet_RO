import random as rd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm
import math
import os


######################################## BASIC FUNCTIONS ########################################################################################################################


def get_budget(parent,cost_matrix):
    """Return the budget of a parent"""
    budget = 0
    for field in parent:
        budget += int(cost_matrix[field[0]][field[1]])
    return budget


def get_random_from_list(list1,list2):
    """"return a random elemnt from list1 that is not in list2"""
    element = rd.choice(list1)
    while element in list2:
        element = rd.choice(list1)
    return element


def budget_check(child,free_fields,cost_matrix,child_budget,budget):
    """Check if the budget is respected after a mutation"""
    if (child_budget > budget):
        child_budget -= int(cost_matrix[child[-1][0]][child[-1][1]])
        child.remove(child[-1])
        child.append(get_random_from_list(free_fields,child))
        child_budget += int(cost_matrix[child[-1][0]][child[-1][1]])
        child_budget = budget_check(child,free_fields,cost_matrix,child_budget,budget)   
    return child_budget


def get_random_from_small_list(list1,list2):
    """"
    Returns a bool indicating if a field from list1 that is not in list2 was found and the field if it was found.
    This method is used for small list1.
    """
    found = False
    found_field = None
    list1_copy = list1[:] 
    rd.shuffle(list1_copy)
    for field in list1_copy:
        if field not in list2:
            found = True
            found_field = field
            break
    return (found,found_field)


def search_for_field(budget, cost_matrix, max_iter, free_fields, child, child_budget):
    """
    Search for a field that is not in the child and add it to the child if the budget is respected.
    Searches for max_iter times.
    """
    child.remove(child[-1])
    for _ in range(max_iter): #a set number of iteration to be able to fill the budget as closely as possible
        field = get_random_from_list(free_fields,child)
        cost = int(cost_matrix[field[0]][field[1]])
        if(child_budget +cost <= budget) :
            child.append(field)
            child_budget += cost
            break
    return child_budget


def search_for_field2(parent, budget, cost_matrix, max_iter, child, child_budget):
    """
    Search for a field in the parent that is not in the child and add it to the child if the budget is respected.
    Searches for max_iter times.
    (This method is used for small parent list)
    """
    child.remove(child[-1])
    for _ in range(max_iter):
        found,field = get_random_from_small_list(parent,child)
        if(not found) : break
        cost = int(cost_matrix[field[0]][field[1]])
        if(child_budget + cost <= budget):
            child.append(field)
            child_budget += cost
            break
    return child_budget


def get_score_lists_index(element,score_list):
    """
    Return the index of the element in the score_list
    method made for the production/habitation/compact score lists
    """
    for i in range(len(score_list)):
        if score_list[i][0] == element:
            return i
    return None


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
    """Return a list of the coordinates of the char in the map_matrix"""
    habitation_list = []
    for i in range(len(map_matrix)):
        for j in range(len(map_matrix[i])):
            if map_matrix[i][j] == char:
                habitation_list.append((i,j))
    return habitation_list


def read_data(cost_file_path,production_file_path,map_file_path):
    """Read the data from the files and return the data in miltiple matrices and lists"""
    cost_matrix = get_matrix(cost_file_path)
    production_matrix = get_matrix(production_file_path)
    map_matrix = get_matrix(map_file_path)
    habitation_list = get_list(map_matrix,'C')
    free_fields = get_list(map_matrix,' ')
    return cost_matrix,production_matrix,map_matrix,habitation_list,free_fields


def write_result(best_solution,seed,gen_length,gen_nbr,best_score,weight_production,weight_habitation,weight_compacity,production_pref,habitation_pref,compacity_pref):
    """Write the results in a csv file"""
    file_path = f'results\\result_{seed}_{gen_length}_{gen_nbr}_{weight_production}_{weight_habitation}_{weight_compacity}_{production_pref}_{habitation_pref}_{compacity_pref}.csv'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write('best solution;score_production;score_habitaion;score_compacity;seed;gen_length;gen_nbr;weight_production;weight_habitation;weight_compacity;prdouction_pref;habitation_pref;compacity_pref\n')
        f.write(f'{best_solution};{best_score[0]};{best_score[1]};{best_score[2]};{seed};{gen_length};{gen_nbr};{weight_production};{weight_habitation};{weight_compacity};{production_pref};{habitation_pref};{compacity_pref}')


######################################## GEN ########################################################################################################################


def generate_parent(free_fields,budget,cost_matrix,max_iter):
    """Generate a parent"""
    parent = []
    parent_budget = 0
    while (parent_budget < budget):
        parent.append(get_random_from_list(free_fields,parent)) #random field added that is not yet in parent
        cost = int(cost_matrix[parent[-1][0]][parent[-1][1]])
        if(parent_budget +cost > budget):
            parent.remove(parent[-1])
            for _ in range(max_iter): #a set number of iteration to be able to fill the budget as closely as possible
                field = get_random_from_list(free_fields,parent)
                cost = int(cost_matrix[field[0]][field[1]])
                if(parent_budget +cost <= budget):
                    parent.append(field)
                    parent_budget += cost
                    break
            break
        else:
            parent_budget += cost
    return parent


def generate_parents(free_fields,budget,cost_matrix,gen_length,max_iter):
    """Generate a list of parents"""
    parents = []
    for _ in range(gen_length):
        parent = generate_parent(free_fields,budget,cost_matrix,max_iter)
        parents.append(parent)
    return parents


def mutate(budget, cost_matrix, max_iter, free_fields, child, child_budget):
    """Mutate a child"""
    field_to_mutate = rd.choice(child)
    child_budget -= int(cost_matrix[field_to_mutate[0]][field_to_mutate[1]]) #removes the cost of mutated field from the budget
    child.remove(field_to_mutate)
    child.append(get_random_from_list(free_fields,child))
    cost = int(cost_matrix[child[-1][0]][child[-1][1]])
    if (child_budget + cost > budget ):
        child_budget = search_for_field(budget, cost_matrix, max_iter, free_fields, child, child_budget)
    else :
        child_budget += cost
    return child_budget


def generate_child(parent1,parent2,budget,cost_matrix,max_iter,mutate_rate,free_fields):
    """Generate a child from two parents"""
    ### INIT ###
    child = []
    child_budget = 0
    parent1_budget = rd.choice(range(10,budget-10)) # Select a random budget that determines the number of fields passed by the 1st parent 

    ### First part of the child ###
    while (child_budget < parent1_budget and len(child) < len(parent1)):
        found,field =get_random_from_small_list(parent1,child) 
        if(not found) : break
        child.append(field)
        child_budget += int(cost_matrix[child[-1][0]][child[-1][1]])

    ### Second part of the child ###
    while (child_budget < budget and len(child) < len(parent1) + len(parent2)):
        found,field = get_random_from_small_list(parent2,child)
        if(not found) : break
        child.append(field)
        cost = int(cost_matrix[child[-1][0]][child[-1][1]])
        if(child_budget + cost > budget):
            combined_parents = parent1 + parent2
            child_budget = search_for_field2(combined_parents, budget, cost_matrix, max_iter, child, child_budget)
            break
        else:
            child_budget += cost

    ### Mutation ###
        if(rd.random() < mutate_rate):
            child_budget = mutate(budget, cost_matrix, max_iter, free_fields, child, child_budget)
            if(rd.random() < mutate_rate):
                child_budget = mutate(budget, cost_matrix, max_iter, free_fields, child, child_budget)
    return child


def generate_new_gen(old_gen,score_list,elite_portion,mutate_rate,budget,cost_matrix,max_iter,free_fields,weights):
    """Generate a new generation from the old one"""
    new_gen = []

    ### Elitism ###
    elite_length = int(len(old_gen)*elite_portion)
    fitness_list = [] #List of tuples (index,fitness)
    production_score_list,average_distance_list,compacity_score_list = score_lists(score_list)
    for i in range(len(old_gen)):
        fitness_list.append((i,calculate_fitness(i,weights,production_score_list,average_distance_list,compacity_score_list)))
    fitness_list.sort(key=lambda x: x[1],reverse=True)
    for i in range(elite_length):
        new_gen.append(old_gen[fitness_list[i][0]])

    ### Crossover ###
    while len(new_gen) < len(old_gen):
        nums = rd.sample(range(0,elite_length),2)
        child = generate_child(old_gen[nums[0]],old_gen[nums[1]],budget,cost_matrix,max_iter,mutate_rate,free_fields)
        new_gen.append(child)
    return new_gen


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
                total_field_distance += math.sqrt((field[0] - habitation_list[j][0])**2 + (field[1] - habitation_list[j][1])**2)
            average_field_distance = total_field_distance / len(habitation_list)
            distance_matrix[field[0]][field[1]] = average_field_distance   
        total_distance += average_field_distance
    average_distance = total_distance / len(parent) # average distance between all the fields and all the habitations
    return average_distance


def production_score(parent,production_matrix):
    """Calculate the production score of a parent"""
    total_production = 0
    for i in range(len(parent)):
        total_production += int(production_matrix[parent[i][0]][parent[i][1]])
    return total_production


def compacity_score(parent):
    """Calculate the compacity score of a parent whitch is the surface of the parent divided by the number of adjacent fields"""
    L = ( max(parent)[0] - min(parent)[0] ) + 1
    l = ( max(parent,key=lambda x:x[1])[1] - min(parent,key=lambda x:x[1])[1] ) + 1
    surface = L*l
    adjacent_score = 1
    for field in parent:
        if((field[0]-1,field[1]) in parent):
            adjacent_score += 1
        if((field[0]+1,field[1]) in parent):
            adjacent_score += 1
        if((field[0],field[1]-1) in parent):
            adjacent_score += 1
        if((field[0],field[1]+1) in parent):
            adjacent_score += 1
    return surface/adjacent_score


def calculate_score(parent,habitation_list,production_matrix,distance_matrix):
    """Calculate the score of a parent, returns a tuple (production,habitation,compacity)"""
    prod_score = production_score(parent,production_matrix)
    habitation_average_distance = average_distance(parent,habitation_list,distance_matrix)
    comp_score = compacity_score(parent)
    score = (prod_score,habitation_average_distance,comp_score)
    return score


def calculate_gen_score(gen,habitation_list,production_matrix,distance_matrix):
    """Calculate the score of a generation, returns a list of tuples (production,habitation,compacity)"""
    score_list = []
    for parent in gen:
        score_list.append(calculate_score(parent,habitation_list,production_matrix,distance_matrix))
    return score_list


def score_lists(score_list):
    """
    Return three lists with the best production, habitation and compacity scores (ordered ).
    Each list is a list of tuples (index,score)
    """
    index_score_list = []
    for i in range(len(score_list)):
        index_score_list.append((i,score_list[i]))
    production_score_list = sorted(index_score_list,key=lambda x:x[1][0])
    habitation_score_list = sorted(index_score_list,key=lambda x:x[1][1],reverse=True)
    compacity_score_list = sorted(index_score_list,key=lambda x:x[1][2],reverse=True)
    return production_score_list,habitation_score_list,compacity_score_list


def calculate_fitness(solution_index,weights,production_score_list,habitation_score_list,compacity_score_list):
    """Calculate the fitness of a parent, returns a float"""
    prod_index = get_score_lists_index(solution_index,production_score_list)
    hab_index = get_score_lists_index(solution_index,habitation_score_list)
    comp_index = get_score_lists_index(solution_index,compacity_score_list)
    fitness = prod_index + hab_index + comp_index #fitness is the sum of the indexes of the solution in the three score lists
    return fitness


######################################## ALGORITHM ########################################################################################################################


def genetic_algorithm(gen_length, gen_nbr, elite_portion, mutate_rate, max_iter, budget, weights, cost_matrix, production_matrix, habitation_list, free_fields, distance_matrix):
    """Main function of the genetic algorithm"""
    ### INIT ###
    gen = generate_parents(free_fields,budget,cost_matrix,gen_length,max_iter)
    score_list = calculate_gen_score(gen,habitation_list,production_matrix,distance_matrix)
    pareto_list = init_pareto(score_list,gen) #list of pareto solutions = (parent,score)

    ### LOOP ###
    for _ in tqdm(range(gen_nbr)):
        ###Genrate the new generation and calculate the score of the new generation ###
        new_gen = generate_new_gen(gen,score_list,elite_portion,mutate_rate,budget,cost_matrix,max_iter,free_fields,weights)
        new_score_list = calculate_gen_score(new_gen,habitation_list,production_matrix,distance_matrix)

        ### Update the pareto list ###
        pareto_list = update_pareto(pareto_list,new_score_list,new_gen)

        ### Update the current generation and score list ###
        gen = new_gen[:]
        score_list = new_score_list[:]
    
    return pareto_list


######################################## PARETO ########################################################################################################################


def init_pareto(score_list,gen):
    """Init the pareto list, returns a list of tuples (parent,(production,habitation,compacity))"""
    pareto_list = []
    for i in range(len(score_list)):
        is_pareto = True
        for solution in pareto_list:
            if(solution[1][0] >= score_list[i][0] and solution[1][1] <= score_list[i][1] and solution[1][2] <= score_list[i][2]):
                is_pareto = False
                break
        if(is_pareto):
            pareto_list.append((gen[i],score_list[i]))
    return pareto_list


def update_pareto(pareto_list,score_list,gen):
    """Update the pareto list, returns a list of tuples (parent,(production,habitation,compacity))"""
    bad_solutions = []
    for i in range(len(score_list)):
        is_pareto = True
        for solution in pareto_list:
            if (solution[1][0] >= score_list[i][0] and solution[1][1] <= score_list[i][1] and solution[1][2] <= score_list[i][2]):
                is_pareto = False
                break
            if (solution[1][0] <= score_list[i][0] and solution[1][1] >= score_list[i][1] and solution[1][2] >= score_list[i][2]):
                if(solution not in bad_solutions):
                    bad_solutions.append(solution)
        if(is_pareto):
            pareto_list.append((gen[i],score_list[i]))

    ### Remove bad solutions ###
    for bad_solution in bad_solutions:
        pareto_list.remove(bad_solution)
    return pareto_list


######################################## PROMETHEE ########################################################################################################################


def promethee(score1,score2,weights,preference):
    """Calculate the promethee score of solution 1 compared to solution 2"""
    score = 0
    for i in range(len(score1)): 
        if(i == 0): #Criteria to maximize (production)
            difference = score1[i] - score2[i]
        else : #Criteria to minimize (habitation and compacity)
            difference = score2[i] - score1[i]
        if(difference > preference[i][0]):
            if(difference < preference[i][1]):
                score += weights[i]*(difference-preference[i][0])/(preference[i][1]-preference[i][0])
            else:
                score += weights[i]
    return score


def create_promethee_list(score_list,weights,preference):
    """Create the promethee list"""
    promethee_list =[[0 for i in range(len(score_list))] for j in range(len(score_list))]
    for i in range(len(score_list)):
        promethee_list[i][i] = 0
        for j in range(len(score_list)):
            if(i != j):
                promethee_list[i][j] = promethee(score_list[i],score_list[j],weights,preference)
                promethee_list[j][i] = promethee(score_list[j],score_list[i],weights,preference)
    return promethee_list


def calculate_promethee_score(score_list,weights,preference):
    """Calculate the promethee score of a generation, returns a sorted list of tuples (index,score)"""
    promethee_score_list = []
    promethee_list = create_promethee_list(score_list,weights,preference)
    for i in range(len(promethee_list)):
        positive_score = sum(promethee_list[i]) / len(promethee_list)
        negative_score = 0
        for j in range(len(promethee_list)):
            negative_score += promethee_list[j][i]
        negative_score /= len(promethee_list)
        promethee_score_list.append((i,positive_score-negative_score))
    return sorted(promethee_score_list,key=lambda x:x[1],reverse=True)


######################################## PLOT ########################################################################################################################


def show_map (map_matrix,solution,scores,gen_length,seed,gen_nbr,weights):
    """Show the map"""
    ### Channge map_matrix to show the solution ###
    cmap = plt.get_cmap('winter')
    bounds = ['C', 'R', ' ','S']
    norm = plt.Normalize(vmin=0, vmax=3)

    ### Convert the matrix to a NumPy array of chr indices ###
    indices = {c: i for i, c in enumerate(bounds)}
    data = np.array([[indices['S'] if (line, col) in solution else indices[map_matrix[line][col]] for col in range(len(map_matrix[line]))] for line in range(len(map_matrix))])

    ### Create a heatmap using Matplotlib ###
    fig, ax = plt.subplots()
    heatmap = ax.imshow(data, cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Map')

    ### ajoute un texte dans lequel on peut indiquer divers informations ###
    text =  f"""La production est de {scores[0]},
    la distance aux habitations est {scores[1]:.2f} et
    le score de compacité {scores[2]:.2f}.
    Avec une population de {gen_length} et {seed} comme rd.seed.
    En {gen_nbr} générations.
    Avec {weights} comme poids (p,h,c)."""
    ax.text(0.5, -0.1,text,transform=ax.transAxes,ha='center', va='center')


def plot_graphs(coordinates,map_matrix,best_solution,best_solution_score,gen_length,seed,gen_nbr,weights):
    """Plot the graphs"""
    ### INTI ###
    filtered_coordinates = []
    for coordinate in coordinates:
        if(coordinate[2] < 50):
            filtered_coordinates.append(coordinate)
    if (len(filtered_coordinates) > 0):
        x,y,z = zip(*filtered_coordinates)
    else:
        x,y,z = zip(*coordinates)
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d',xlabel=f'Production',ylabel='Habitation',zlabel='Compacity')
    ax2 = fig.add_subplot(122, projection='3d',xlabel='Production',ylabel='Habitation',zlabel='Compacity')
    
    ### PARETO GRAPH ###
    ax.scatter(x, y, z, c=z,alpha=1)
    ax.set_title('Pareto front')

    ### SURFACE GRAPH ###
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='nearest')
    zi = np.clip(zi, a_min=0, a_max=None)
    ax2.plot_surface(xi, yi, zi)
    ax2.set_title('Surface')

    ### SHOW MAP ###
    show_map(map_matrix,best_solution,best_solution_score,gen_length,seed,gen_nbr,weights)


######################################## MAIN ########################################################################################################################


def main():
    ### Parameters ###
    # Seed #
    seed = rd.randint(0,1000000)
    rd.seed(seed)
    # Genetic algorithm parameters #
    gen_length = 2000  #number of parents in a generation
    gen_nbr = 1 #number of generations
    elite_portion = 0.5  #portion of the best parents that will be kept in the next generation
    mutate_rate = 0.5  #rate of mutation
    max_iter = 15  #maximum number of tries to find a field that fits the budget
    budget = 50
    # Files #
    cost_file_path = 'donnes_V2\\Cost_map.txt'
    production_file_path = 'donnes_V2\\Production_map.txt'
    map_file_path = 'donnes_V2\\Usage_map.txt'
    # Promethee parameters #
    weights = (0.7,0.3,0.5) #weights of the criteria (production,habitation,compacity)
    preference = ((2,15),(0.5,6),(0.1,10)) #preference of the criteria (min,max) (production,habitation,compacity)

    ### Data ###
    cost_matrix, production_matrix, map_matrix, habitation_list, free_fields = read_data(cost_file_path, production_file_path, map_file_path)
    distance_matrix = [[-1 for _ in range(len(map_matrix[j]))] for j in range(len(map_matrix))] # a matrix to store the average distance between a field and the habitations

    ### Genetic Algorithm ###
    pareto_list = genetic_algorithm(gen_length, gen_nbr, elite_portion, mutate_rate, max_iter, budget, weights, cost_matrix, production_matrix, habitation_list, free_fields, distance_matrix)
    solutions,scores = zip(*pareto_list)

    ### Promethee ###
    promethee_score_list = calculate_promethee_score(scores,weights,preference)
    best_solution = solutions[promethee_score_list[0][0]]
    best_solution_score = scores[promethee_score_list[0][0]]

    ### Write the solution in a file ###
    write_result(best_solution,seed,gen_length,gen_nbr,best_solution_score,weights[0],weights[1],weights[2],preference[0],preference[1],preference[2])

    ### Plot ###
    plot_graphs(scores,map_matrix,best_solution,best_solution_score,gen_length,seed,gen_nbr,weights)
    plt.show()


if __name__ == '__main__':
    main()