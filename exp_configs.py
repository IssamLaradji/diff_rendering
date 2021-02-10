from haven import haven_utils as hu

EXP_GROUPS = {}
EXP_GROUPS['project1'] = pass
EXP_GROUPS['project2'] = pass
EXP_GROUPS['project3'] = pass
EXP_GROUPS['project4'] = pass
EXP_GROUPS['project5'] = pass

EXP_GROUPS['moved_block'] =  [
    # {
    #         "source_scene": 
    #                       {'name':'moved_block'},
    #         "target_scene": 
    #                       {'name':'basic'},
    #         "model": 
    #                       {'name':'basic_renderer', 'opt':'sps'},
    #      },
         {
            "source_scene": 
                          {'name':'moved_block'},
            "target_scene": 
                          {'name':'basic'},
            "model": 
                          {'name':'basic_renderer', 'opt':'adam', 'lr':5e-1},
         }]


# EXP_GROUPS = {k: hu.cartesian_exp_group(v) for k, v in EXP_GROUPS.items()}