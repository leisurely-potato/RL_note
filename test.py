import gym

env = gym.make('CartPole-v1', render_mode='human')  # 改成v1，并加render_mode
state, _ = env.reset()

for _ in range(1000):
    env.render()
    print(state)
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        state = env.reset()
        

env.close()
