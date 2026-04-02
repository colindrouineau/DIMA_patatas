# AI detection of blight on potatoe leaves.
2 months research project

## Build a new Neural Network model on a new type of data

IN :
### CONFIG
- Add necessary entries in CONFIG file (TRAINING_INFO, DATA_TYPE, MODEL_NAME : any parameter you need and want to be able to change easily. It MUST contain at least those 3 entries : LEARNING_RATE, NUM_EPOCHS, DELTA)
- Change TRAINING_CHOICE in CONFIG file to match the training you want to proceed

### train_nn 
- Import your nn_model
- add a function : def define_{model_type}_{data_type}_functions, which initialises as attributes the useful functions for training : criterion, optimizer, ... You may want to add Parameters for these functions in CONFIG file
- add a case in `define_nn_functions`
- in nn_results function, you may want to add a case for your specific model.

### nn_models 
- Create a class to Define the architecture of the model in algo.nn_models.py. name model. There should be a `save_nn` and a `forward` function. Choose an appropriate name for the model.

### test_models 
- add a case in performance if you have a new data_type (if you can't use one that already exists)
- define a performance function :
def perff(y_val, y_pred): print(info on model perf); return metric dictionary ("metrics" : metrics)
- add a case in `load_model`

### open_image
- add function {datatype}array that loads and returns the image array (2D) of your chosen datatype

### format_data
- in leaf_mask_data : add a case for your new data type, open it, and creates "mask_lab" which selects only desired pixels. PROBLEM : MAKES IT 1D. For CNN, what do we do? All my code works with pixels. I should think about it.
- in load_data : add a case if the data needs specific further formatting.
- in load_data_verbose : add a case for verbosing your new data_type when it's loaded.




## Operation done on original dataset:
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