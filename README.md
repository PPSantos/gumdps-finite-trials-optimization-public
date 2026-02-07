# Solving General-Utility Markov Decision Processes in the Single-Trial Regime with Online Planning

https://arxiv.org/abs/2505.15782

## Files overview:
* `simulate_mcts` contains the code to run the MCTS algorithm for the illustrative GUMDPs.
* `simulate_mcts_gym` contains the code to run the MCTS algorithm for the Taxi and FrozenLake environments.
* `simulate_mcts_gym_continuous` contains the code to run the MCTS algorithm for the MountainCar environment.
* The notebooks containing the code to produce the plots of the article (as well as to run the baselines) are: `exps-mcts.ipynb` (Illustrative GUMDPs), `exps-mcts-gym-envs.ipynb` (Taxi and FrozenLake envs.), and `exps-mcts-gym-envs-continuous.ipynb` (MountainCar env.).

## Environments:

#### Illustrative GUMDPs:
* $\mathcal{M}_{f,1}$, max. state entropy exploration: `entropy_mdp_10`
* $\mathcal{M}_{f,2}$, imitation learning: `imitation_learning_mdp_10`
* $\mathcal{M}_{f,3}$, adversarial MDP: `adversarial_mdp_10`

#### Gym environments:
* FrozenLake, max. state entropy exploration: `frozen_lake_entropy`
* FrozenLake, imitation learning: `frozen_lake_imitation_learning`
* Taxi, max. state entropy exploration: `taxi_entropy`
* Taxi, imitation learning: `taxi_imitation_learning`
* MountainCar, max. state entropy exploration: `mountain_car_continuous_entropy`
* MountainCar, imitation learning: `mountain_car_continuous_imitation_learning`


Tested with Python 3.8.10
