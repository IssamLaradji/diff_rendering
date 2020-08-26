from haven import haven_utils as hu

EXP_GROUPS = {}
EXP_GROUPS['moved_block'] =  {
            "source_scene": 
                          {'name':'moved_block'},
            "target_scene": 
                          {'name':'basic'},
            "model": 
                          {'name':'basic_renderer', 'opt':'adam', 'lr':5e-1},
         }


EXP_GROUPS = {k: hu.cartesian_exp_group(v) for k, v in EXP_GROUPS.items()}