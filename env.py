import gym

ENV_NAME = 'shooting'
NUM_EPISODES = 12000
NO_OP_STEPS = 30

env = gym.make(ENV_NAME)
agent = Agent(num_actions=env.action_space.n)

observation = env.start()
#学習スタート
for _ in xrange(NUM_EPISODES):
    terminal = False #エピソード終了判定を初期化
    for _ in xrange(random.randint(1, NO_OP_STEPS))#ランダムフレーム分何もしない
        last_observation = observation
        observation, _, _ = env.step(0)

    state = agent.get_initial_state(observation, last_observation)#初期状態作成

    while not terminal:
        last_observation = observation
        action = agent.get_action(state)#行動選択
        observation, reward = env.step(action)#実行,次の画像と報酬が帰ってくる
        state = agent.run(state, action, reward, terminal, processed_observation)#学習
