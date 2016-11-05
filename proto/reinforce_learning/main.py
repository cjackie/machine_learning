import abc


# class VariableSpace:

#     @abstractmethod
#     def getAllNumbers(self):
#         '''
#         @return, list, all state numbers
#         '''
#         pass;

#     @abstractmethod
#     def getName(self, number):
#         '''
#         @stateNumber, an item from getAllStateNumber.
#         @return, string representing the state.
#         '''
#         pass;


# everything is hardcoded.
states = [0,1,2,3,4,5]
actions = ['0','1']
reward = {0: 0, 1: -2.0, 2: -3.0, 3: -1.0, 4: -2.0, 5: -2.0}
def transitionProbabiltyDistribution(state, action):
    '''
    @return a list, represents probiblity distribution like
        [0, 0, 0.3, 0.5, 0, 0].
    '''
    p = {
        (0, '0'): [1,0,0,0,0,0],
        (0, '1'): [0.6,0.4,0,0,0,0],
        (1, '0'): [0.2,0.4,0,0,0,0.8],
        (1, '1'): [0.3,0.2,0,0,0,0.5],
        (3, '0'): [0,0.2,0.1,0.5,0.2,0],
        (3, '1'): [0,0.5,0,0.5,0,0],
        (5, '0'): [0.1,0,0.9,0,0,0],
        (5, '1'): [0,0,0.7,0.3,0,0],
        (2, '0'): [0,0,0,0.8,0,0.2],
        (2, '1'): [0,0.1,0.9,0,0,0],
        (4, '0'): [0,0,0.2,0,0.8,0],
        (4, '1'): [0,0,0.1,0.1,0.8,0]
    }
    return p[(state,action)]

policy = {}

def snapshotPolicy(policy):
    s = "=========== policy =============\n"
    s += "policy is \n"
    for state, action in policy.items():
        s += "at state " + str(state) + " do: " + action + "\n"
    print(s)

def snapshotValues(values):
    s = "=========== values =============\n"
    s += "values is: \n"
    for state, value in values.items():
        s += "at state " + str(state) + " value: " + str(value) + "\n"
    print(s)


def valueIterationAlgorithm(states, actions, reward, transitionProbabiltyDistribution):
    iterNum = 10

    #initialize optimal values
    optimalValues = {}
    optimalPolicy = {}
    for s in states:
        optimalValues[s] = 0.0

    # probability expectation of valueFunction(s) given action
    def valueExpectation(valueFunction,givenAction,givenState):
        value = 0.0
        distribution = transitionProbabiltyDistribution(givenState, givenAction)
        for s in states:
            value += distribution[s]*valueFunction[s]
        return value

    discount = 0.99    
    for i in range(iterNum):
        for s in states:
            # update optimals for each state
            values = {}
            for action in actions:
                values[action] = reward[s] + discount* \
                    valueExpectation(optimalValues, action, s)
            action,maxValue = max(values.items(), key=lambda x: x[1])
            optimalValues[s] = maxValue
            # snapshotValues(optimalValues)
            optimalPolicy[s] = action
        snapshotPolicy(optimalPolicy)
    return (optimalPolicy, optimalValues)


optimalPolicy, _ = valueIterationAlgorithm(states, actions, reward, transitionProbabiltyDistribution)
# snapshotPolicy(optimalPolicy)


class State:
    @abstractmethod
    def uid(self):
        '''
        @return :int, an unique identifier of the state.
        '''
        pass

    @abstractmethod
    def __str__(self):
        '''
        @return :string, string representation of the state
        '''

    def __eq__(self, state):
        '''
        @state :State
        @return :bool, check if two state are equal.
        '''
        return state.uid() == self.uid()

class StateSpace:
    '''
    The finite state space
    '''

    def __init__(self):
        self._stateSpace = []

    def addState(self, state):
        '''
        add a state to the space
        @state: State
        '''
        for s in self._stateSpace:
            if s == state:
                raise Exception("state " + str(state) + " exists")
        self._stateSpace.add(s)

    def __item__

    def __itr__
