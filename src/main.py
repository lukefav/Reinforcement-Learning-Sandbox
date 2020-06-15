import argparse
from argparse import RawTextHelpFormatter

from q_learning.q_learning import QLearning


def main():
    help_description = "Choose Project:\n" \
                       "0: QLearning"
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

    else:
        print(f"Incorrect project number, you entered: {project_num}")


if __name__ == "__main__":
    main()
