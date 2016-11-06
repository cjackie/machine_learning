import abc
from scipy.stats import rv_discrete

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




##################################################################################


class StateSpace:

    def __init__(self, states):
        '''
        @states :array of int.
        '''
        self._states = states[:]

    def states(self):
        return self._states[:]

    def __iter__(self):
        return self._states.__iter__()

class ActionSpace:
    def __init__(self, actions):
        '''
        @actions :array of int
        '''
        self._actions = actions[:]

    def __iter__(self):
        return self._actions.__iter__()

    def actions(self):
        return self._actions[:]


class TransitionProbability:
    def __init__(self, actionSpace, stateSpace):
        '''
        @actionSpace :ActionSpace
        @stateSpace :StateSpace
        '''
        self._actionSpace = actionSpace
        self._actionSpaceSet = set(actionSpace)
        self._stateSpace = stateSpace
        self._stateSpaceSet = set(stateSpace)
        self._randomVariables = {}
        for s in stateSpace.states():
            for a in actionSpace.actions():
                self._randomVariables[(s,a)] = None

    def update(self, givenState, givenAction, pk):
        '''
        @givenAction :int, action in the actionSpace
        @giveState: int, state in the stateSpace
        @pk: dict, key is state in stateSpace
            items are positive probability with sum(pk) = 1
        '''
        # checking inputs
        if givenAction not in self._actionSpaceSet:
            raise Exception("wrong argument")

        if givenState not in self._stateSpaceSet:
            raise Exception("wrong argument")

        probSum = 0.0
        xk = []
        pk = []
        if state, prob in pk.items():
            if state not in self._stateSpaceSet:
                raise Exception("wrong argument")
            probSum += prob
            xk.append(state)
            pk.append(prob)
        if probSum != 1.0:
            raise Exception("wrong argument")

        # update
        self._randomVariables[(state, action)] = scipy.stats.rv_discrete(values=(xk, pk))

    def __setitem__(self, given, pk):
        '''
        wrapper of update
        @given :(int state, int action), tuple of (state, action).
        '''
        self.update(given[0], given[1], pk)


    def get(self, givenState, givenAction):
        return self._randomVariables[(givenState, givenAction)]

    def __getitem__(self, given):
        '''
        @given :(int state, int action), tuple of (state, action).
        @return, scipy.stats.rv_discrete.
        '''
        return self.get(given[0], given[1])

    def ready(self):
        '''
        check if all action and state has a pmf
        @return :bool
        '''
        for t, rv in self._randomVariables:
            if rv == None:
                return False
        return True


class StateReward:

    def __init__(self, stateSpace, rewards):
        '''
        @stateSpace :StateSpace
        @reward: dict, key represent state and value represents
                how much reward.
        '''
        # make sure @rewards covers all states in stateSpace
        keys = rewards.key()
        states = stateSpace()
        if len(keys) != len(states):
            raise Exception("error")
        keys = sorted(keys)
        states = sorted(states)
        for i in range(len(keys)):
            if keys[i] != states[i]:
                raise Exception("error")

        self._rewards = dict(rewards)

    def rewards():
        return dict(rewards)

    def __getitem__(self, key):
        return self._rewards[key]



class Policy:
    '''
    map of state => action.
    '''
    def __init__(self, actionSpace, stateSpace)
        self._actionSpaceSet = set(actionSpace.states())
        self._stateSpaceSet = set(stateSpace.actions())
        self._policy = {}
        for s in stateSpace:
            self._policy[s] = None;

    def update(self, state, action):
        '''
        update policy
        '''
        if state not in self._stateSpaceSet or \
            action not in self._actionSpaceSet:
            raise Exception("error")
        self._policy[state] = action

    def policyAt(self, state):
        '''
        @return int, an action in actionSpace. None if no policy
        '''
        if state not in self._stateSpaceSet:
            raise Exception("error")

        return self._policy[state]

    def policy(self):
        return dict(self._policy)


class ValueIterationAlgorithm:
    def __init__(self, stateSpace, actionSpace, rewards, transitionProbability, discount):
        '''
        stateSpace :StateSpace
        actionSpace :ActionSpace
        rewards :StateReward
        transitionProbability: TransitionProbability, it need to be ready.
        discount :double, discount factor.
        '''
        if not transitionProbability.ready():
            raise "not ready"

        self.stateSpace = stateSpace
        self.actionSpace = actionSpace
        self.rewards = rewards
        self.transitionProbability = transitionProbability
        self.discount = discount


    def run(self, numIter=None, verbose=False):
        if numIter == None:
            # run until convergence
            raise Exception("not yet implemented")
        else:
            optimalValues = {}
            optimalPolicy = Policy()
            for s in self.stateSpace:
                optimalValues[s] = 0.0

            for i in range(numIter):
                for state in self.stateSpace:
                    values = {}
                    for action in self.actionSpace:
                        rv = self.transitionProbability[(state, action)]
                        f = lambda s: optimalValues[s]
                        values[action] = reward[state] + discount*rv.expect(func=f)

                    action, maxValue = max(values.items(), key=lambda x: x[1])
                    optimalValues[s] = maxValue
                    optimalPolicy.update(state, action)

                if verbose:
                    print("at iteration number " + str(i) + ":\n")
                    self._snapshotPolicyAndValue(optimalValues, optimalPolicy)
            return optimalPolicy


    def _snapshotPolicyAndValue(self, optimalValues, optimalPolicy):
        print("=========== optimal values ================\n")
        for state, value in optimalValues.items():
            print("state " + str(state) + " => " + "value "+ str(value) + "\n")
        print("=========== optimal policy ================\n")
        for state, action in optimalPolicy.policy().items():
            print("state " + str(state) + " action " + str(action) + "\n")
        print("\n")


def usageExample():
    # TODO


