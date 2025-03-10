import gym_MPR
import gymnasium


def main():
    env_name  = "MatPodRacer-v0"
    env = gymnasium.make(env_name)

    print(env.action_space.n)

main()