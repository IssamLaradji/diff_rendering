
import exp_configs
import argparse
import pandas as pd

from src import scenes
from src import models

from haven import haven_utils as hu
import os, pprint



def trainval(exp_dict):
    pprint.pprint(exp_dict)

    savedir_base = os.path.join('tmp', hu.hash_dict(exp_dict))
    os.makedirs(savedir_base, exist_ok=True)
    # -- get scenes
    source_scene = scenes.get_scene(exp_dict['source_scene'])
    target_scene = scenes.get_scene(exp_dict['target_scene'])

    # -- get model
    model = models.get_model(exp_dict['model'], source_scene, exp_dict)

    # -- train for E iterations
    score_list = []
    for e in range(500):
        # update parameters and get new score_dict
        score_dict = model.train_on_batch(target_scene)
        score_dict["epoch"] = e
        score_dict["step_size"] = model.opt.state['step_size']

        # Add to score_list and save checkpoint
        score_list += [score_dict]

        # Print 
        score_df = pd.DataFrame(score_list)
        print("\n", score_df.tail(), "\n")

        # Visualize
        if e % 50 == 0:
            model.vis_on_batch(target_scene, 
                    fname=os.path.join(savedir_base, 
                            'output_%d.png' % e))

    save_gif(src_path=os.path.join(savedir_base, '*.png'), 
             tgt_fname=s.path.join(savedir_base, 'animation.gif'))


def save_gif(src_path, tgt_fname):
    import glob
    from PIL import Image

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(src_path))]
    img.save(fp=tgt_fname, format='GIF', append_images=imgs,
            save_all=True, duration=200, loop=0)

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


    