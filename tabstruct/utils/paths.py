
import os 
import json

def find_checkpoint(base_path, encoding_type, task, logger=None):
    """Find checkpoint in the specified task and encoding type directories."""
    models_path = os.path.join(base_path, 'models', encoding_type, task)
    #logger.info(f"Searching for checkpoints in: {models_path}")
    try:
        paths = os.listdir(models_path)
        checkpoint = [path for path in paths if path.startswith('checkpoint')]
        if len(checkpoint) == 1:
            return os.path.join(models_path, checkpoint[0])
        else:
            if logger is not None:
                logger.warning(f"Multiple or no checkpoints found: {checkpoint}")
            return None
    except FileNotFoundError:
        #logger.error(f"No directory found for: {models_path}")
        return None
    



def get_max_checkpoint(model_args, logger):
    # get save results path : 

    save_dir = f"../logs/results/{model_args.task}"
    save_path =  os.path.join(save_dir, model_args.encoding_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logger.info(f'filename : {model_args.encoding_type}')
    logger.info(f'save_path : {save_path}')


    if os.path.exists( f"{save_path}/validation_{model_args.encoding_type}.json"):
        with open( f"{save_path}/validation_{model_args.encoding_type}.json", 'r') as f:
            data = json.load(f)
        
        max_checkpoint = max(data, key=lambda checkpoint: data[checkpoint]['accuracy'])

    else:
        max_checkpoint = None  

    return max_checkpoint, save_path