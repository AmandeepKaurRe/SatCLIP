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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from main import SatCLIPLightningModule

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


model_path = "/Users/gmuhawen/Gedeon/Research/geossl/downstream/checkpoints/UAR.ckpt"
continent_names = ["Africa", "Oceania","North_america", "South_america", "Asia", "Europe", "Global"]
# continent_names = ["Global", "Africa"]
models = {
    "Zero":"-",
    "UAR":"UAR",
    "Stratified_Continents":"continent",
    "Stratified_Biome":"biome",
    "SeCo":"seco",
    "Biased_Americentric":"biased_americentric",
    "Biased_Forestly":"biased_forestly"   
}
n_runs = 10
downstream_model_type = "KNN"


def get_air_temp_data(continent_name, pred="temp", norm_y=True, norm_x=True):
    continent_name = continent_name.lower()
    file_path = f"/Users/gmuhawen/Gedeon/Research/geossl/Air-Temperature-Downstream/QGIS/data/{continent_name}_air_temp.shp"
    gdf = gpd.read_file(file_path)
    
    coords = gdf[['Lon', 'Lat']].to_numpy(dtype=np.float32) 
    
    # Choose prediction target based on 'pred' parameter
    if pred == "temp":
        y = gdf['meanT'].to_numpy(dtype=np.float32)  
        x = gdf['meanP'].to_numpy(dtype=np.float32) 
    else:
        y = gdf['meanP'].to_numpy(dtype=np.float32) 
        x = gdf['meanT'].to_numpy(dtype=np.float32)  

    # Normalize if required
    if norm_y:
        y = y / y.max() if y.max() != 0 else y 
    if norm_x:
        x = x / x.max() if x.max() != 0 else x 

    return torch.tensor(coords), torch.tensor(x), torch.tensor(y)


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

  train_size = int(0.6 * total_size)
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

def sample_train_data(x_train, y_train, n_samples):
    # Initialize a random number generator with a different seed each time
    rng = np.random.default_rng()  # Automatically uses a random seed
    
    # Randomly sample indices from the range of x_train
    indices = rng.choice(len(x_train), size=n_samples, replace=False)
    
    # Index into x_train and y_train using the sampled indices
    x_sample = x_train[indices]
    y_sample = y_train[indices]
    
    return x_sample, y_sample

class MLP(nn.Module):
    def __init__(self, input_dim, dim_hidden, num_layers, out_dims):
        super(MLP, self).__init__()

        layers = []
        layers += [nn.Linear(input_dim, dim_hidden, bias=True), nn.ReLU()] # Input layer
        layers += [nn.Linear(dim_hidden, dim_hidden, bias=True), nn.ReLU()] * num_layers # Hidden layers
        layers += [nn.Linear(dim_hidden, out_dims, bias=True)] # Output layer

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)

def train_random_forest(x_train, y_train, x_test, n_estimators=100, max_depth=None):
    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_model.fit(x_train, y_train)  # Train the model
    
    test_predictions = rf_model.predict(x_test)  # Predict on test set
    return rf_model, test_predictions

def train_knn(x_train, y_train, x_test, n_neighbors=5):
    knn_model = KNeighborsRegressor(n_neighbors=n_neighbors, metric='euclidean')
    knn_model.fit(x_train, y_train)  # Train the model
    
    test_predictions = knn_model.predict(x_test)  # Predict on test set
    return knn_model, test_predictions

def train_model(model, criterion, optimizer, x_train, y_train, x_test, device='cpu', epochs=3000, print_interval=250):
    """
    Trains a model using mean squared error loss and Adam optimizer.

    Parameters:
    - model: PyTorch model to be trained
    - criterion: Loss function
    - optimizer: Optimizer
    - x_train: Training data features (torch.Tensor)
    - y_train: Training data labels (torch.Tensor)
    - x_test: Test data features (torch.Tensor)
    - device: Device to perform training ('cpu' or 'cuda')
    - epochs: Number of training epochs
    - print_interval: Interval at which to print the loss

    Returns:
    - model: Trained PyTorch model
    - losses: List of training losses
    - y_pred_test: Model predictions on the test data
    """
    model = model.float().to(device)
    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        y_pred = model(x_train.float().to(device))
      
        loss = criterion(y_pred.reshape(-1), y_train.float().to(device))
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())

        if (epoch + 1) % print_interval == 0:
            pass
            # print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        y_pred_test = model(x_test.float().to(device))
    
    return model, losses, y_pred_test

all_resulsts = {}
samples = {}
results_csv = {
                "Pre-training model":[],
                "Continent":[],
                "Number of Finetuning samples":[],
                "Test MSE":[],
                "Finetuning model type":[]
                }
samples_range = [10, 20, 30, 40, 50, 100, 200, 400, 800, 1600]
for continent in continent_names:
    for model_name in models.keys():
        print(f"Running Downstream for {continent}, with {model_name} pretraining")

        coords, _, y = get_air_temp_data(continent_name=continent)
        model_path = f"/Users/gmuhawen/Gedeon/Research/geossl/downstream/checkpoints/{models[model_name]}.ckpt"
        if model_name=="Zero":
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

                if downstream_model_type=="Random Forest":
                    x_train_samples_np = x_train_samples.cpu().numpy()
                    y_train_samples_np = y_train_samples.cpu().numpy()
                    x_test_np = x_test.cpu().numpy()
                    y_test_np = y_test.cpu().numpy()

                    # Train RandomForest model
                    rf_model, test_predictions = train_random_forest(x_train_samples_np, y_train_samples_np, x_test_np)
                    test_error = mean_squared_error(y_test_np, test_predictions)
                    print(f'RF - {continent} with {n_samples} samples has a Test loss of: {test_error}, data size{y_train_samples_np.shape}')

                if downstream_model_type=="KNN":
                    x_train_samples_np = x_train_samples.cpu().numpy()
                    y_train_samples_np = y_train_samples.cpu().numpy()
                    x_test_np = x_test.cpu().numpy()
                    y_test_np = y_test.cpu().numpy()

                    # Train KNN model
                    knn_model, test_predictions = train_knn(x_train_samples_np, y_train_samples_np, x_test_np)

                    # Calculate test error using MSE
                    test_error = mean_squared_error(y_test_np, test_predictions)
                    print(f'KNN - {continent} with {n_samples} samples has a Test loss of: {test_error}, data size{y_train_samples_np.shape}')

                if downstream_model_type=="MLP":
                    pred_model = MLP(input_dim=256, dim_hidden=64, num_layers=2, out_dims=1).float().to(device)
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(pred_model.parameters(), lr=0.001)
                    trained_model, training_losses, test_predictions = train_model(pred_model, criterion, optimizer, x_train_samples, y_train_samples, x_test, device=device)
                    test_error = criterion(test_predictions.reshape(-1), y_test.float().to(device)).item()
                    print(f'MLP - {continent} with {n_samples} samples has a Test loss of: {test_error}, data size{y_train_samples.shape}')


                # Write our results for csv ready format
                results_csv["Pre-training model"].append(model_name)
                results_csv["Continent"].append(continent)
                results_csv["Number of Finetuning samples"].append(n_samples)
                results_csv["Test MSE"].append(test_error)
                results_csv["Finetuning model type"].append(downstream_model_type)

            if n_samples >= max_samples:
                samples_range.remove(max_samples)
                break

    #     break #remove
    # break #remove
df = pd.DataFrame(results_csv)
df.to_csv(f'Air_Temperature_Results_{downstream_model_type}.csv', index=False)

    

#   plot_true_vs_predicted(coords_test, y_test, test_predictions, continent_name=continent)
