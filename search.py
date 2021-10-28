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
from copy import deepcopy

import util
from game import Directions


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
    return [s, s, w, s, w, w, s, w]


def getNextPaths(problem, currentPath, expandedStates: list or None) -> (list, bool):
    """
    currentPath: list of successors in order
    Returns a tuple (nextPaths, goalReached)

    If goal reached, it means the 'currentPath' is a path that reaches the goal
    """

    # 1. Check if last successor of path reached the goal and return None, True if so
    # 2.

    nextPaths = list()

    if currentPath is None:
        currentPath = list()
        stateToExpand = problem.getStartState()
    else:
        stateToExpand = currentPath[-1][0]

    # Check if the state of the last successor of the path is the goal
    if problem.isGoalState(stateToExpand):
        return list(), True

    if stateToExpand not in expandedStates:
        for successor in problem.getSuccessors(stateToExpand):
            nextPath = deepcopy(currentPath)
            nextPath.append(successor)
            nextPaths.append(nextPath)

        expandedStates.append(stateToExpand)

    return nextPaths, False


def graphSearch(problem, fringe):
    actions = list()
    expandedStates = list()

    nextPaths, foundGoal = getNextPaths(problem, None, expandedStates)

    for nextPath in nextPaths:
        fringe.push(nextPath)

    while not (fringe.isEmpty() or foundGoal):

        currentPath = fringe.pop()

        nextPaths, foundGoal = getNextPaths(problem, currentPath, expandedStates)

        if foundGoal:
            actions = [i[1] for i in currentPath]
        else:
            for nextPath in nextPaths:
                fringe.push(nextPath)

    return actions


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    fringe = util.Stack()
    return graphSearch(problem, fringe)


def breadthFirstSearch(problem):
    fringe = util.Queue()
    return graphSearch(problem, fringe)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    """Search the shallowest nodes in the search tree first."""

    # PriorityQueueWithFunction takes the one with the lowest priority (e.g cost) first, instead of the
    # highest
    fringe = util.PriorityQueueWithFunction(lambda x: problem.getCostOfActions([i[1] for i in x]))
    return graphSearch(problem, fringe)


def getPathCost(path: list):
    """
    Calculates the cumulative path cost to get to destination. Used by the uniformCostSearch
    """

    edgeCosts = [successor[2] for successor in path]
    return sum(edgeCosts)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
