# imports
import argparse
import logging
import os
from PIL import Image
import torch
import torchvision
from torch import Tensor, nn
from torchvision import transforms

from azureml.core import Run, Dataset, Workspace, Model
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.run import _OfflineRun


logger = None
model = None
device = None

classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
def init():
    global logger
    global model
    global device

    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--logging_level', type=str, help='logging level')
    
    for k, v in os.environ.items():
        print(f'{k}={v}')
    models_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model")

   #ws = retrieve_workspace()
    
    model_path = os.path.join(models_path, "cifar_net.pt")
    #model_path=model_get.download(target_dir='.', exist_ok=False, exists_ok=None)
    # parse args
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    logger.setLevel(args.logging_level.upper())

    logger.info('Init started')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Device: %s', device)

    print(" directory ", model_path)
    logger.info(" directory ", model_path)
    # load model
    model = torch.load(model_path, map_location='cpu')
    model.to(device)
    model.eval()

    logger.info('Init completed')


def run(mini_batch):
    logger.info('run(%s started: %s', mini_batch, {__file__})
    predicted_names = []
    transform = transforms.ToTensor()

    for image_path in mini_batch:
        image = Image.open(image_path)
        tensor = transform(image).to(device)
        predicted = predict(model, tensor).item()
        predicted_names.append(f'{image_path}: {predicted}')
    logger.info('Run completed')
    return predicted_names

def predict(trained_model: nn.Module, x: Tensor):
    with torch.no_grad():
        y_prime = trained_model(x)
        predicted = torch.max(y_prime.data, 1)
        #probabilities = nn.functional.softmax(y_prime, dim=1)
        #predicted_indices = probabilities.argmax(1)
    return predicted

def retrieve_workspace() -> Workspace:

    ws = None

    try:
        run = Run.get_context()
        if not isinstance(run, _OfflineRun):
            ws = run.experiment.workspace
            return ws
    except Exception as ex:
        print('Workspace from run not found', ex)

    try:
        ws = Workspace.from_config()
        return ws
    except Exception as ex:
        print('Workspace config not found in local folder', ex)

    try:
        sp = ServicePrincipalAuthentication(
            tenant_id=os.environ['AML_TENANT_ID'],
            service_principal_id=os.environ['AML_PRINCIPAL_ID'],
            service_principal_password=os.environ['AML_PRINCIPAL_PASS']
        )
        ws = Workspace.get(
            name="<ml-example>",
            auth=sp,
            subscription_id="<your-sub-id>"
        )
    except Exception as ex:
        print('Workspace config not found in project', ex)

    return ws

def get_model(ws, model_name, model_version=None, model_path=None):
    """Get or create a compute target.

    Args:
        ws (Workspace): The Azure Machine Learning workspace object
        model_name (str): The name of ML model
        model_version (int): The version of ML model (If None, the function returns latest model)
        model_path (str): The path to a model file (including file name). Used to load a model from a local path.

    Returns:
        Model/model object: The trained model (if it is already registered in AML workspace,
                               then Model object is returned. Otherwise, a model object loaded with
                               joblib is returned)

    """
    model = None

    try:
        model = Model(ws, name=model_name, version=model_version)
        print(f'Found the model by name {model_name} and version {model_version}')
        return model
    except Exception:
        print((f'Cannot load a model from AML workspace by model name {model_name} and model_version {model_version}. '
               'Trying to load it by name only.'))
    try:
        models = Model.list(ws, name=model_name, latest=True)
        if len(models) == 1:
            print(f'Found the model by name {model_name}')
            model = models[0]
            return model
        elif len(models) > 1:
            print('Expected only one model.')
        else:
            print('Empty list of models.')
    except Exception:
        print((f'Cannot load a model from AML workspace by model name {model_name}. '
               'Trying to load it from a local path.'))

    try:
        model = joblib.load(model_path)
        print('Found the model by local path {}'.format(model_path))
        return model
    except Exception:
        print('Cannot load a model from {}'.format(model_path))

    if model is None:
        print('Cannot load a model. Exiting.')
        sys.exit(-1)

    return model