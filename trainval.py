
import exp_configs
import argparse
import pandas as pd

from src import scenes
from src import models




def trainval(exp_dict):
    # -- get scenes
    source_scene = scenes.get_scene(exp_dict['source_scene'])
    target_scene = scenes.get_scene(exp_dict['target_scene'])

    # -- get model
    model = models.get_model(exp_dict['model'], source_scene, exp_dict)

    # -- train for E iterations
    score_list = []
    for e in range(300):
        # update parameters and get new score_dict
        score_dict = model.train_on_batch(target_scene)
        score_dict["epoch"] = e

        # Add to score_list and save checkpoint
        score_list += [score_dict]

        # Print 
        score_df = pd.DataFrame(score_list)
        print("\n", score_df.tail(), "\n")

        # Visualize
        model.vis_on_batch(target_scene, fname='results/output.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")

    args = parser.parse_args()

    # select exp group
    exp_list = []
    for exp_group_name in args.exp_group_list:
        exp_list += exp_configs.EXP_GROUPS[exp_group_name]
    
    for exp_dict in exp_list:
        # do trainval
        trainval(exp_dict)


    