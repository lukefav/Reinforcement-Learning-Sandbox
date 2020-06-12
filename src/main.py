import argparse
from argparse import RawTextHelpFormatter

from q_learning.q_learning import QLearning


def main():
    help_description = "Choose Project:\n" \
                       "1: QLearning"

    parser = argparse.ArgumentParser(description="Choose Reinforcement Learning Project",
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument("project", metavar='p', type=int,
                        help=help_description)

    project_num = parser.parse_args().project

    if project_num == 1:
        print("QLearning Loading...")
        project = QLearning()
        print("Running Project.")
        project.run()

    else:
        print("Incorrect project number")


if __name__ == "__main__":
    main()
