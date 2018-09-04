import numpy as np

class HMM():
    def __init__(self, transition_prob, emission_prob, initial_state_prob):
        '''
            transition probabilitys : 2d nparray
            emission probabilitys: 2d nparray
            initial state prob: 1d nparray
        '''
        self._transition_prob = transition_prob
        self._emission_prob = emission_prob
        self._initial_state_prob = initial_state_prob
        self._state_num = len(transition_prob)
 
    def evaluation_by_forward(self, observation):
        '''
            observation:1d nparray
        '''
        state_num = self._state_num
        emission_prob = self._emission_prob
        transition_prob = self._transition_prob
        initial_state_prob = self._initial_state_prob
        max_t = len(observation)
        α = np.zeros((max_t, state_num))
        for t in range(max_t):
            for j in range(state_num):
                if t == 0 :
                    α[t][j] = initial_state_prob[j] * emission_prob[j][observation[t]]
                else:
                    α[t][j] = sum(α[t-1][i] * transition_prob[i][j] for i in range(state_num)) * emission_prob[j][observation[t]]
    
        return sum(α[max_t-1])

    def decode_by_viterbi_algorithm(self, observation):
        state_num = self._state_num
        emission_prob = self._emission_prob
        transition_prob = self._transition_prob
        initial_state_prob = self._initial_state_prob
        max_t = len(observation)
        δ = np.zeros((max_t, state_num))
        #ψ = np.zeros((max_t, state_num))
        for t in range(max_t):
            for j in range(state_num):
                if t == 0 :
                    δ[t][j] = initial_state_prob[j] * emission_prob[j][observation[t]]
                else:                    
                    δ[t][j] = max(δ[t-1][i] * transition_prob[i][j] for i in range(state_num)) * emission_prob[j][observation[t]]
                    #ψ[t][j] = np.argmax([δ[t-1][i] * transition_prob[i][j] for i in range(state_num)])
    
        return [np.argmax([δ[t][i] for i in range(state_num)]) for t in range(max_t)]

transition_prob = np.asarray([[0.6, 0.5, 0.4], [0.2, 0.3, 0.1], [0.2, 0.2, 0.5]])
emission_prob = np.asarray([[0.7, 0.1, 0.2], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4]])
initial_state_prob = np.asarray([0.5, 0.2, 0.3])

model = HMM(transition_prob, emission_prob, initial_state_prob)

observation = np.asarray([0, 0, 2, 1, 2, 1, 0])
print(model.evaluation_by_forward(observation))
print(model.decode_by_viterbi_algorithm(observation))
'''
double backward(int* o, int T)
{
    for (int t=T-1; t>=0; --t)
        for (int i=0; i<N; ++i)
            if (t == T-1)
                β[t][i] = 1.0;
            else
            {
                double p = 0;
                for (int j=0; j<N; ++j)
                    p += a[i][j] * b[j][o[t+1]] * β[t+1][j];
                β[t][i] = p;
            }
 
    double p = 0;
    for (int j=0; j<N; ++j)
        p += π[j] * b[j][o[0]] * β[0][j];
    return p;
}
'''