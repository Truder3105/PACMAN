# -*- coding: utf-8 -*-

# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Realiza una búsqueda en profundidad (DFS) para encontrar una solución al problema.

    Utiliza una pila (Stack) como estructura de datos para gestionar los nodos a explorar (frontera).
    Este algoritmo explora los nodos en profundidad, y utiliza un conjunto (set) para rastrear los 
    nodos visitados y evitar re-explorar los mismos estados.

    Parámetros:
    - problem: El problema que se está resolviendo. Debe tener métodos:
        - getStartState: Retorna el estado inicial.
        - isGoalState: Retorna True si un estado es objetivo.
        - getSuccessors: Retorna una lista de sucesores para un estado dado (estado, acción, costo).

    Retorna:
    - Una lista de acciones que lleva al estado objetivo, o una lista vacía si no se encuentra solución.
    """

    # Importar la pila desde util
    from util import Stack

    # Inicialización de la frontera con el nodo inicial del problema
    frontier = Stack()
    frontier.push((problem.getStartState(), []))  # (estado actual, acciones realizadas)

    # Inicialización del conjunto de estados visitados
    visited = set()

    # Mientras la frontera no esté vacía
    while not frontier.isEmpty():
        # Extrae el nodo (estado, acciones) de la frontera
        state, actions = frontier.pop()

        # Si es un estado objetivo, retorna las acciones
        if problem.isGoalState(state):
            return actions

        # Si el estado no ha sido visitado
        if state not in visited:
            # Marcar el estado como visitado
            visited.add(state)

            # Expandir los nodos sucesores y agregarlos a la frontera
            for successor, action, cost in problem.getSuccessors(state):
                if successor not in visited:
                    frontier.push((successor, actions + [action]))

    # Si no se encuentra solución, retornar una lista vacía
    return []


def breadthFirstSearch(problem):
    """
    Implementa la búsqueda en amplitud para encontrar la ruta más corta en términos de pasos desde el estado inicial hasta el objetivo.
    
    Args:
        problem: El problema de búsqueda que define los estados, acciones y el objetivo.
    
    Returns:
        Una lista de acciones que lleva al estado objetivo, o una lista vacía si no se encuentra una solución.
    """

    # Utiliza una cola para gestionar los estados por explorar (borde/frontera)
    from util import Queue
    frontier = Queue()

    # Inicia la frontera con el estado inicial y un camino vacío
    start_state = problem.getStartState()
    frontier.push((start_state, []))

    # Conjunto para rastrear los estados ya explorados
    explored = set()

    # Bucle para explorar los nodos hasta que se encuentre la solución o se agoten los nodos
    while not frontier.isEmpty():
        # Desencola el estado más antiguo (FIFO)
        current_state, path = frontier.pop()

        # Retorna el camino si el estado actual es el objetivo
        if problem.isGoalState(current_state):
            return path

        # Si el estado no ha sido explorado previamente
        if current_state not in explored:
            # Marca el estado como explorado
            explored.add(current_state)

            # Genera y explora los sucesores del estado actual
            for successor, action, step_cost in problem.getSuccessors(current_state):
                if successor not in explored:
                    new_path = path + [action]
                    frontier.push((successor, new_path))

    # Retorna una lista vacía si no se encuentra una solución
    return []



def uniformCostSearch(problem):
    """
    Realiza la búsqueda de costo uniforme para encontrar la ruta de menor costo desde el estado inicial hasta el objetivo.
    
    Args:
        problem: Instancia del problema de búsqueda que define estados, acciones y objetivos.
    
    Returns:
        Una lista de acciones que lleva al estado objetivo, o una lista vacía si no se encuentra solución.
    """

    # Cola de prioridad para manejar los estados según su costo acumulado (g(n))
    frontier = util.PriorityQueue()

    # Estado inicial y configuración inicial del costo acumulado
    initial_state = problem.getStartState()
    frontier.push((initial_state, [], 0), 0)  # (estado, ruta, costo acumulado)

    # Diccionario para rastrear los costos mínimos encontrados por estado
    cost_map = {}

    # Bucle principal para explorar los nodos hasta que se encuentre una solución o se agoten las opciones
    while not frontier.isEmpty():
        # Extraer el estado con el costo acumulado más bajo
        current_state, path, accumulated_cost = frontier.pop()

        # Si se ha alcanzado el objetivo, devuelve la secuencia de acciones
        if problem.isGoalState(current_state):
            return path

        # Explora el nodo solo si no ha sido visitado o se encuentra un camino más barato
        if current_state not in cost_map or accumulated_cost < cost_map[current_state]:
            cost_map[current_state] = accumulated_cost  # Actualiza el costo mínimo para este estado

            # Expande los sucesores del estado actual
            for successor, action, step_cost in problem.getSuccessors(current_state):
                # Calcula el nuevo costo acumulado para el sucesor
                total_cost = accumulated_cost + step_cost
                new_path = path + [action]

                # Añade el sucesor a la cola de prioridad con su costo total actualizado
                frontier.push((successor, new_path, total_cost), total_cost)
    
    # Si no se encuentra una solución, devuelve una lista vacía
    return []



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """
    Implementa el algoritmo de búsqueda A* para encontrar la ruta óptima en un espacio de estados.
    
    Args:
        problem: El problema de búsqueda que define los estados, acciones y objetivos.
        heuristic: Función heurística que estima el costo restante desde un estado dado hasta el objetivo.
    
    Returns:
        Una lista de acciones que lleva al estado objetivo, o una lista vacía si no se encuentra solución.
    """

    # Cola de prioridad para gestionar los estados según su costo total estimado (f(n) = g(n) + h(n))
    frontier = util.PriorityQueue()

    # Estado inicial del problema y su costo asociado (g(n) = 0)
    start_state = problem.getStartState()
    frontier.push((start_state, [], 0), heuristic(start_state, problem))

    # Diccionario para rastrear los costos mínimos encontrados para cada estado
    explored = {}

    # Bucle principal para expandir nodos hasta que se encuentre una solución o se agoten las opciones
    while not frontier.isEmpty():
        # Extrae el estado con el costo total estimado más bajo
        current_state, path, path_cost = frontier.pop()

        # Si el estado actual es el objetivo, devuelve la secuencia de acciones que llevó a este estado
        if problem.isGoalState(current_state):
            return path

        # Si el estado no se ha explorado antes o se ha encontrado una ruta más barata
        if current_state not in explored or path_cost < explored[current_state]:
            explored[current_state] = path_cost  # Marca el estado como explorado con el costo asociado

            # Expande los sucesores del estado actual
            for successor, action, step_cost in problem.getSuccessors(current_state):
                # Calcula el costo acumulado para llegar al sucesor
                total_cost = path_cost + step_cost
                new_path = path + [action]

                # Calcula el valor de f(n) = g(n) + h(n) para el sucesor y lo agrega a la cola de prioridad
                frontier.push((successor, new_path, total_cost), total_cost + heuristic(successor, problem))

    # Si no se encuentra una solución, devuelve una lista vacía
    return []




# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
