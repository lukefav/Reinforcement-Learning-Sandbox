import argparse
from argparse import RawTextHelpFormatter

from deep_q_learning.deep_q_learning import DeepQLearning
from q_learning.q_learning import QLearning


def main():
    help_description = "Choose Project:\n" \
                       "0: QLearning,\n" \
                       "1: DeepQLearning"
    parser = argparse.ArgumentParser(description="Choose Reinforcement Learning Project",
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument("project", metavar='p', type=int,
                        help=help_description)

    project_num = parser.parse_args().project

    if project_num == 0:
        print("QLearning Loading...")
        project = QLearning()
        print("Running Project.")
        project.run()

    elif project_num == 1:
        print("DeepQLearning Loading...")
        project = DeepQLearning()
        print("Running Project.")
        project.run()

    else:
        print(f"Incorrect project number, you entered: {project_num}")


if __name__ == "__main__":
    main()
