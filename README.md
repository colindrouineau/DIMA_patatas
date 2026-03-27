# AI detection of blight on potatoe leaves.
2 months research project

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_PST_2023
Cuda compilation tools, release 12.0, V12.0.140
Build cuda_12.0.r12.0/compiler.32267302_0



## Operation on original dataset:
    """
    # Cuts that are already made. (for intruder leaf)
    
    PATH = "/home/colind/work/Mines/TR_DIMA/DIMA_code/data/Lab_Feb2025_Mask/foliolo3/enves/foliolo3_enves_a8.png"
    P1 = (131, 56)
    P2 = (223, 61)

    img_cleaner.cut_in_line(PATH, P1, P2, "left")

    PATH = "/home/colind/work/Mines/TR_DIMA/DIMA_code/data/Lab_Feb2025_Mask/foliolo3/enves/foliolo3_enves_a5.png"
    P1 = (161, 57)
    P2 = (220, 61)

    img_cleaner.cut_in_line(PATH, P1, P2, "left")

    P1 = (223, 73)
    P2 = (232, 96)

    PATH = "/home/colind/work/Mines/TR_DIMA/DIMA_code/data/Lab_Feb2025_Mask/foliolo5/enves"
    files = os.listdir(PATH)
    for file in files:
        path = os.path.join(PATH, file)
        img_cleaner.cut_in_line(path, P1, P2, "left", inplace=True)
"""

    # PATH = "/home/colind/work/Mines/TR_DIMA/DIMA_code/data/Lab_Feb2025_Mask_arch/foliolo2/enves/foliolo2_enves_a5.png"
    #
    # img_cleaner.cut_stem_image(PATH)

    # FOLDER = None
    # img_cleaner.cut_allstems()