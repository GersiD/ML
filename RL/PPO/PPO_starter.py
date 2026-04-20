import torch
import gymnasium as gym
import datetime
import matplotlib.pyplot as plt

from RolloutBuffer import RolloutBuffer

class ActorCritic(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Softmax(dim=-1)
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def act(self, state):
        """
        Given the current state query the actor network for the action to take.
        Returns:
            *action*: the action to take
            *action_logprobs*: the log probability of the action
            *value*: the expected return of our policy starting from this state
        """
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_logprobs = dist.log_prob(action)
        value = self.critic(state)
        return action.detach(), action_logprobs.detach(), value.detach()

    def evaluate(self, state, action):
        """
        Given the current state and action which was sampled previously, evaluate the actor and critic networks.
        Returns:
            *action_logprobs*: the log probability of the action
            *value*: the expected return of our policy starting from this state
            *dist_entropy*: the entropy of the action distribution
        """
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        value = self.critic(state).squeeze()
        return action_logprobs, value, dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, ) -> None:
        # Hyperparameters
        self.lr_actor = 0.002 # learning rate for actor
        self.lr_critic = 0.001 # learning rate for critic
        self.gamma = 0.99 # discount factor
        self.eps_clip = 0.2 # clip parameter for PPO
        self.epochs = 10 # number of training epochs
        self.hidden_dim = 64 # hidden dimension for both actor and critic networks
        self.buffer = RolloutBuffer() # replay buffer
        self.policy = ActorCritic(state_dim, action_dim, self.hidden_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
        ])
        # because we need to calculate the ratio of the new policy to the old policy
        self.policy_old = ActorCritic(state_dim, action_dim, self.hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state, device=device)
            action, action_logprobs, value = self.policy_old.act(state)
        self.buffer.insert(state, action, action_logprobs, value)
        return action.item()

    def update(self):
        """
        Update the policy using the PPO algorithm.
        Uses the experiences stored in the buffer to update the policy.
        """

        # TODO: Implement the PPO update algorithm using the experiences stored in self.buffer

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
        self.policy_old.load_state_dict(torch.load(path))

def main():
    print("============================================================================================")
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    if env is None:
        print(f"Environment {env_name} not found")
        return
    print("Environment: " + env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    ppo = PPO(state_dim, action_dim)
    print("============================================================================================")
    print(f"Training started at {datetime.datetime.now()}")
    print("============================================================================================")
    max_episodes = 500
    max_steps = 1000
    update_timestep = 2000
    timestep = 0
    training_reward = []
    for i_episode in range(1, max_episodes+1):
        state = env.reset()[0]
        episode_reward = 0
        for _ in range(max_steps):
            timestep += 1
            action = ppo.act(state)
            state, reward, done, _, _ = env.step(action)
            ppo.buffer.rewards.append(reward)
            ppo.buffer.is_terminals.append(done)
            episode_reward += float(reward)
            if timestep % update_timestep == 0:
                # print(f"---Update---")
                ppo.update()
            if done:
                break
        # print(f"Episode: {i_episode}, Reward: {episode_reward}")
        training_reward.append(episode_reward)
    print("============================================================================================")
    print(f"Training finished at {datetime.datetime.now()}")
    print("============================================================================================")
    env.close()
    ppo.save("ppo.pth")
    print("Model saved successfully")
    plt.plot(training_reward)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO Training Reward")
    plt.savefig("ppo_training_reward.png")


if __name__ == "__main__":
    # set device to cpu or cuda
    device = torch.device('cpu')
    if(torch.cuda.is_available()): 
        device = torch.device('cuda:0') 
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
    print("============================================================================================")
    main()
