import torch
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from torch.utils.data import TensorDataset, random_split
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split

from urllib import request
import numpy as np
import pandas as pd
import io
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from main import SatCLIPLightningModule

continent_names = ["Africa", "Oceania", "North_america", "South_america", "Asia", "Europe", "Global"]
# continent_names = ["Africa"]

models = {
    # "Zero": "-",
    # "UAR": "UAR",
    # "Stratified_Continents": "continent",
    # "Stratified_Biome": "biome",
    # "SeCo": "seco",
    # "Biased_Americentric": "biased_americentric",
    # "Biased_Forestly": "biased_forestly",
    "Biased_Asia": "biased_asia"
}

n_runs = 10
downstream_model_type = "KNN"

print(f"Runing for : {downstream_model_type}")

def get_eco_region_data(continent_name, norm_x=True):
    continent_name = continent_name.lower()
    file_path = f"//Users/gmuhawen/Gedeon/Research/geossl/biome_downstream/data/{continent_name}_eco_recion.shp"
    gdf = gpd.read_file(file_path)
    coords = gdf[['lon', 'lat']].to_numpy(dtype=np.float32)

    if 'biome_clas' not in gdf.columns:
        raise ValueError("The 'biome_clas' column is missing in the dataset.")
    
    gdf['biome_clas'] = gdf['biome_clas'].replace(-1, 0)
    y = gdf['biome_clas'].to_numpy(dtype=np.float32)

    return torch.tensor(coords), torch.tensor(y)

def plot_true_vs_predicted(coords_test, y_test, y_pred_test, continent_name):
    """
    Plot the true and predicted data on two side-by-side subplots and save the plot to a file.
    
    Parameters:
        coords_test (numpy.ndarray): Coordinates of the test data.
        y_test (numpy.ndarray): True labels for the test data.
        y_pred_test (numpy.ndarray): Predicted labels for the test data.
        output_file (str): Path to save the output image file.
    """

    output_file = continent_name + '.png'

    if continent_name:
        lon_min, lat_min, lon_max, lat_max = get_continent_bounds(continent_name)
    else:
        # Calculate bounding box based on coordinates if no continent is specified
        lon_min, lat_min = coords_test.min(axis=0)
        lon_max, lat_max = coords_test.max(axis=0)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))

    # Plotting the 'True' map
    m = Basemap(projection='cyl', llcrnrlon=lon_min, llcrnrlat=lat_min, 
                urcrnrlon=lon_max, urcrnrlat=lat_max, resolution='c', ax=ax[0])
    
    # fig, ax = plt.subplots(1, 2, figsize=(10, 3))

    # # Plotting the 'True' map
    # m = Basemap(projection='cyl', resolution='c', ax=ax[0])
    m.drawcoastlines()
    ax[0].scatter(coords_test[:, 0], coords_test[:, 1], c=y_test, s=5)
    ax[0].set_title('True')

    # Plotting the 'Predicted' map
    m = Basemap(projection='cyl', resolution='c', ax=ax[1])
    m.drawcoastlines()
    ax[1].scatter(coords_test[:, 0], coords_test[:, 1], c=y_pred_test.reshape(-1), s=5)
    ax[1].set_title('Predicted')

    # Save the figure to a file
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

def get_continent_bounds(continent_name):
    """
    Returns the bounding box (lon_min, lat_min, lon_max, lat_max) for the given continent.
    
    Parameters:
        continent_name (str): The name of the continent.
        
    Returns:
        tuple: The bounding box (lon_min, lat_min, lon_max, lat_max) of the continent.
    """
    continents = {
        'Africa-': (-17.5, -35, 51.5, 37.5),
        'Europe-': (-25, 34, 45, 71),
        'Asia': (25, -10, 180, 81),
        'North_america-': (-169, 5, -52, 84),
        'South_america-': (-82, -56, -34, 13),
        'Oceania-': (112, -45, 155, -10)
    }
    return continents.get(continent_name, (-180, -90, 180, 90))  # Default to world view if unknown continent



def load_satCLIP(ckpt_path=None, device='cpu', return_all=False):
    if ckpt_path is not None:
        # Load model from checkpoint
        print("Loading the checkpoints")
        ckpt = torch.load(ckpt_path, map_location=device)
        lightning_model = SatCLIPLightningModule(**ckpt['hyper_parameters']).to(device)
        lightning_model.load_state_dict(ckpt['state_dict'])
    else:
        # Initialize model with random parameters
        print("Using Zeroshot")
        lightning_model = SatCLIPLightningModule()  # Pass any required arguments for model initialization

    lightning_model.eval()
    geo_model = lightning_model.model

    if return_all:
        return geo_model.eval
    else:
        return geo_model.location
    
def encode_locations(coords, model_path):
  model = load_satCLIP(model_path, device=device) # Only loads location encoder by default
  model.eval()
  with torch.no_grad():
    x  = model(coords.double().to(device)).detach().cpu()
  return x


def train_valid_test_split(coords, embeddings, y, random_seed=0):
  dataset = TensorDataset(coords, embeddings, y)

  total_size = len(dataset)

  train_size = int(0.5 * total_size)
  valid_size = int(0.0 * total_size)
  test_size = total_size - train_size - valid_size

  if random_seed is not None:
    np.random.seed(random_seed)

  # Create indices
  indices = list(range(total_size))
  np.random.shuffle(indices)

  # Split indices into train, validation, and test sets
  train_indices = indices[:train_size]
  valid_indices = indices[train_size:train_size + valid_size]
  test_indices = indices[train_size + valid_size:]

  TRAIN = dataset.tensors[0][train_indices], dataset.tensors[1][train_indices], dataset.tensors[2][train_indices]
  VALID = dataset.tensors[0][valid_indices], dataset.tensors[1][valid_indices], dataset.tensors[2][valid_indices]
  TEST = dataset.tensors[0][test_indices], dataset.tensors[1][test_indices], dataset.tensors[2][test_indices]

  return TRAIN, VALID, TEST

def remap_classes_to_zero_based(tensor):
    unique_classes = torch.unique(tensor)
    class_mapping = {old_class.item(): new_class for new_class, old_class in enumerate(unique_classes)}
    remapped_tensor = torch.tensor([class_mapping[c.item()] for c in tensor])
    return remapped_tensor

def sample_train_data(x_train, y_train, n_samples):
    # Initialize a random number generator with a different seed each time
    rng = np.random.default_rng()  # Automatically uses a random seed
    
    # Randomly sample indices from the range of x_train
    indices = rng.choice(len(x_train), size=n_samples, replace=False)
    
    # Index into x_train and y_train using the sampled indices
    x_sample = x_train[indices]
    y_sample = y_train[indices]
    
    return x_sample, y_sample

def train_random_forest(x_train, y_train, x_test, n_estimators=100, max_depth=None):
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_model.fit(x_train, y_train)
    test_predictions = rf_model.predict(x_test)
    return rf_model, test_predictions


def train_knn(x_train, y_train, x_test, n_neighbors=5):
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
    knn_model.fit(x_train, y_train)
    test_predictions = knn_model.predict(x_test)
    return knn_model, test_predictions

def train_skmlp(x_train, y_train, x_test, hidden_layer_sizes=(100,), max_iter=200, activation='relu', solver='adam'):

    mlp_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, activation=activation, solver=solver, random_state=42)
    mlp_model.fit(x_train, y_train)
    test_predictions = mlp_model.predict(x_test)
    
    return mlp_model, test_predictions



def train_model(model, criterion, optimizer, x_train, y_train, x_test, device='cpu', epochs=3000, print_interval=250):
    model = model.float().to(device)
    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        y_pred = model(x_train.float().to(device))

        loss = criterion(y_pred, y_train.long().to(device))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % print_interval == 0:
            pass

    model.eval()
    with torch.no_grad():
        y_pred_test = model(x_test.float().to(device))
    
    return model, losses, y_pred_test

class MLP(nn.Module):
    def __init__(self, input_dim, dim_hidden, num_layers, out_dims):
        super(MLP, self).__init__()

        layers = []

        layers.append(nn.Linear(input_dim, dim_hidden, bias=True))
        layers.append(nn.ReLU())

        for _ in range(num_layers):
            layers.append(nn.Linear(dim_hidden, dim_hidden, bias=True))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dim_hidden, out_dims, bias=True))

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)

all_results = {}
samples = {}
results_csv = {
    "Pre-training model": [],
    "Continent": [],
    "Number of Finetuning samples": [],
    "Test Accuracy": [],
    "F1 Score": [],
    # "ROC AUC": [],
    "Finetuning model type": []
}
samples_range = [10, 20, 30, 40, 50, 100, 200, 400, 800, 1600, 3200, 6500]
for continent in continent_names:
    for model_name in models.keys():
        print(f"Running Downstream for {continent}, with {model_name} pretraining")

        coords, y = get_eco_region_data(continent_name=continent)
        model_path = f"/Users/gmuhawen/Gedeon/Research/geossl/downstream/checkpoints/{models[model_name]}.ckpt"
        if model_name == "Zero":
            model_path = None
        embeddings = encode_locations(coords, model_path)
        TRAIN, VALID, TEST = train_valid_test_split(coords, embeddings, y, random_seed=0)
        coords_train, x_train, y_train = TRAIN
        coords_valid, x_valid, y_valid = VALID
        coords_test, x_test, y_test = TEST

        max_samples = len(y_train)
        samples_range.append(max_samples)

        for n_samples in sorted(samples_range):
            run_ = 0
            while run_ < n_runs:
                run_ += 1
                x_train_samples, y_train_samples = sample_train_data(x_train, y_train, n_samples)

                if downstream_model_type == "RF":
                    x_train_samples_np = x_train_samples.cpu().numpy()
                    y_train_samples_np = y_train_samples.cpu().numpy()
                    x_test_np = x_test.cpu().numpy()
                    y_test_np = y_test.cpu().numpy()

                    # Train RandomForestClassifier
                    rf_model, test_predictions = train_random_forest(x_train_samples_np, y_train_samples_np, x_test_np)

                    # Calculate metrics
                    test_predictions_np = test_predictions
                    test_accuracy = accuracy_score(y_test_np, test_predictions_np)
                    test_f1 = f1_score(y_test_np, test_predictions_np, average='weighted')
                    
                    # Binarize labels for multi-class ROC AUC calculation
                    # y_test_bin = label_binarize(y_test_np, classes=np.unique(y_test_np))
                    # test_roc_auc = roc_auc_score(y_test_bin, test_predictions_np, average='weighted', multi_class='ovr')


                if downstream_model_type == "KNN":
                    x_train_samples_np = x_train_samples.cpu().numpy()
                    y_train_samples_np = y_train_samples.cpu().numpy()
                    x_test_np = x_test.cpu().numpy()
                    y_test_np = y_test.cpu().numpy()

                    # Train KNNClassifier
                    knn_model, test_predictions = train_knn(x_train_samples_np, y_train_samples_np, x_test_np)

                    # Calculate metrics
                    test_predictions_np = test_predictions
                    test_accuracy = accuracy_score(y_test_np, test_predictions_np)
                    test_f1 = f1_score(y_test_np, test_predictions_np, average='weighted')

                if downstream_model_type == "SKMLP":
                    x_train_samples_np = x_train_samples.cpu().numpy()
                    y_train_samples_np = y_train_samples.cpu().numpy()
                    x_test_np = x_test.cpu().numpy()
                    y_test_np = y_test.cpu().numpy()

                    knn_model, test_predictions = train_skmlp(x_train_samples_np, y_train_samples_np, x_test_np)

                    # Calculate metrics
                    test_predictions_np = test_predictions
                    test_accuracy = accuracy_score(y_test_np, test_predictions_np)
                    test_f1 = f1_score(y_test_np, test_predictions_np, average='weighted')


                if downstream_model_type == "MLP":
                    out_dim = torch.unique(y_train_samples).numel()
                    y_train_samples = remap_classes_to_zero_based(y_train_samples)
                    # print(y_train_samples)
                    pred_model = MLP(input_dim=256, dim_hidden=64, num_layers=2, out_dims=out_dim).float().to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(pred_model.parameters(), lr=0.001)
                    trained_model, training_losses, test_predictions = train_model(pred_model, criterion, optimizer, x_train_samples, y_train_samples, x_test, device=device)

                    # Calculate metrics
                    test_predictions_np = np.argmax(test_predictions.cpu().numpy(), axis=1)
                    test_predictions_np = np.argmax(test_predictions, axis=1)
                    y_test_np = y_test.cpu().numpy()
                    test_accuracy = accuracy_score(y_test_np, test_predictions_np)
                    test_f1 = f1_score(y_test_np, test_predictions_np, average='weighted')
                
                print(f'{downstream_model_type} - {continent} with {n_samples} samples:')
                print(f'  Test Accuracy: {test_accuracy}')
                print(f'  F1 Score: {test_f1}')
                # print(f'  ROC AUC: {test_roc_auc}')

                # Write results to CSV format
                results_csv["Pre-training model"].append(model_name)
                results_csv["Continent"].append(continent)
                results_csv["Number of Finetuning samples"].append(n_samples)
                results_csv["Test Accuracy"].append(test_accuracy)
                results_csv["F1 Score"].append(test_f1)
                # results_csv["ROC AUC"].append(test_roc_auc)
                results_csv["Finetuning model type"].append(downstream_model_type)
    
            if n_samples >= max_samples:
                samples_range.remove(max_samples)
                break
    
    df = pd.DataFrame(results_csv)
    df.to_csv(f'AS_Eco_regions_Classification_Results_{downstream_model_type}.csv', index=False)

df = pd.DataFrame(results_csv)
df.to_csv(f'AS_Eco_regions_Classification_Results_{downstream_model_type}.csv', index=False)
