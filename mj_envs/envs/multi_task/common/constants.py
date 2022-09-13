import numpy as np

# ToDo: Get these details from key_frame
DEMO_RESET_QPOS = np.array(
    [
        1.01020992e-01,
        -1.76349747e00,
        1.88974607e00,
        -2.47661710e00,
        3.25189114e-01,
        8.29094410e-01,
        1.62463629e00,
        3.99760380e-02,
        3.99791002e-02,
        2.45778156e-05,
        2.95590127e-07,
        2.45777410e-05,
        2.95589217e-07,
        2.45777410e-05,
        2.95589217e-07,
        2.45777410e-05,
        2.95589217e-07,
        2.16196258e-05,
        5.08073663e-06,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        -2.68999994e-01,
        3.49999994e-01,
        1.61928391e00,
        6.89039584e-19,
        -2.26122120e-05,
        -8.87580375e-19,
    ]
)

DEMO_RESET_QVEL = np.array(
    [
        -1.24094905e-02,
        3.07730486e-04,
        2.10558046e-02,
        -2.11170651e-02,
        1.28676305e-02,
        2.64535546e-02,
        -7.49515183e-03,
        -1.34369839e-04,
        2.50969693e-04,
        1.06229627e-13,
        7.14243539e-16,
        1.06224762e-13,
        7.19794728e-16,
        1.06224762e-13,
        7.21644648e-16,
        1.06224762e-13,
        7.14243539e-16,
        -1.19464428e-16,
        -1.47079926e-17,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        2.93530267e-09,
        -1.99505748e-18,
        3.42031125e-14,
        -4.39396125e-17,
        6.64174740e-06,
        3.52969879e-18,
    ]
)

OBJ_INTERACTION_SITES = (
    "knob1_site",
    "knob2_site",
    "knob3_site",
    "knob4_site",
    "light_site",
    "slide_site",
    "leftdoor_site",
    "rightdoor_site",
    "microhandle_site", 
    "kettle_site",
    "kettle_site",
    "kettle_site",
    "kettle_site",
    "kettle_site",
    "kettle_site",
)

OBJ_JNT_NAMES = (
    "knob1_joint",
    "knob2_joint",
    "knob3_joint",
    "knob4_joint",
    "lightswitch_joint",
    "slidedoor_joint",
    "leftdoorhinge",
    "rightdoorhinge",
    "micro0joint",
    "kettle0:Tx",
    "kettle0:Ty",
    "kettle0:Tz",
    "kettle0:Rx",
    "kettle0:Ry",
    "kettle0:Rz",
)

ROBOT_JNT_NAMES = (
    "panda0_joint1",
    "panda0_joint2",
    "panda0_joint3",
    "panda0_joint4",
    "panda0_joint5",
    "panda0_joint6",
    "panda0_joint7",
    "panda0_finger_joint1",
    "panda0_finger_joint2",
)

TEXTURE_ID_TO_INFOS = {
    1: dict(
        name='floor', 
        shape=(1024, 1024, 3), 
        group='floor',
    ),
    5: dict(
        name='sink_handle', 
        shape=(512, 512, 3),
        group='handle',
    ),
    6: dict(
        name='sink_top', 
        shape=(512,512,3),
        group='surface',
    ),
    7: dict(
        name='drawer', 
        shape=(512,512,3),
        group='surface',
    ),
    10: dict(
        name='sdoor_handle', 
        shape=(512,512,3),
        group='handle',
    ),
    11: dict(
        name='sdoor_surface', 
        shape=(512,512,3),
        group='surface',
    ),
    12: dict(
        name='lrdoor_surface', 
        shape=(512,512,3),
        group='surface',
    ),
    13: dict(
        name='lrdoor_handle', 
        shape=(512, 512, 3),
        group='handle',
    ),
    14: dict(
        name='micro_handle', 
        shape=(512, 512, 3),
        group='handle',
    ),
    
    16: dict(
        name='kettle_handle', 
        shape=(512, 512, 3),
        group='handle',
    ),
}


OBJ_JNT_RANGE = {
    'lightswitch_joint': (-0.6, 0), 
    'rightdoorhinge': (0, 0.2), 
    'slidedoor_joint': (0, 0.3), 
    'leftdoorhinge': (-0.2, 0), 
    'micro0joint': (-1, 0), 
    'knob1_joint': (-1, 0), 
    'knob2_joint': (-1, 0),   
    'knob3_joint': (-1, 0), 
    'knob4_joint': (-1, 0), 
    'kettle0:Tx': (-0.4, -0.1), #(-0.1, 0.1),
    'kettle0:Ty': (0.1, 0.5), # (0.0, 0.2),
}



DEFAULT_BODY_RANGE = {
            "counters": { # note this includes both left counter and right sink
                "pos": {
                    "center": [0, 0, 0],
                    "low":    [0, -.4, 0],
                    "high":   [0, .4, 0],
                },
                "euler": {
                    "center": [0, 0, 0],
                    "low":    [0, 0, 0],
                    "high":   [0, 0, 0],
                }
            },
            "microwave": {
                "pos": {
                    "center": [-0.750, -0.025, 1.6],
                    "low":    [-.1, -.07, 0],
                    "high":   [0.05, 0.075,0],            
                    },
                "euler": {
                    "center": [0, 0, 0.3],
                    "low":    [0, 0, -.15],
                    "high":   [0, 0, .15],
                },
            },
            "hingecabinet": {
                "pos": {
                    "center": [-0.504, 0.28, 2.6],
                    "low":    [-.1, -.1, 0],
                    "high":   [0, .05, .1],
                },
                "euler": {
                    "center": [0, 0, 0],
                    "low":    [0, 0, 0],
                    "high":   [0, 0, 0],
                },
            },
            "slidecabinet": {
                "pos": {
                    "center":  [0.4, 0.28, 2.6], #None,  # use hingecabinet randomzied pos
                    "low":     [0, 0, 0],
                    "high":    [0.1, 0, 0],
                },
                "euler": {
                    "center": [0, 0, 0],
                    "low":    [0, 0, 0],
                    "high":   [0, 0, 0],
                },
            },
            "kettle": {
                "pos": {
                    "center":  [-0.269, 0.35, 1.626],  
                    "low":     [0, 0, 0],
                    "high":    [0.5, 0.45, 0],
                },
                "euler": {
                    "center": [0, 0, 0],
                    "low":    [0, 0, 0],
                    "high":   [0, 0, 0],
                },
            }
}