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

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from main import SatCLIPLightningModule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


continent_names = ["Africa", "Oceania",
                    "North_america", "South_america", "Asia", "Europe", "Global"
                    ]
models = {
    "Zero": "-",
    "UAR": "UAR",
    "Stratified_Continents": "continent",
    "Stratified_Biome": "biome",
    "SeCo": "seco",
    "Biased_Americentric": "biased_americentric",
    "Biased_Forestly": "biased_forestly"
}

n_runs = 10
downstream_model_type = "MLP"

print(f"Runing for : {downstream_model_type}")

def get_population_data(continent_name):
    continent_name = continent_name.lower()
    file_path = f"/Users/gmuhawen/Gedeon/Research/geossl/Population_Downstream/data/{continent_name}_population.shp"
    gdf = gpd.read_file(file_path)
    coords = gdf[['Lon', 'Lat']].to_numpy(dtype=np.float32)
    
    if 'log_popula' not in gdf.columns:
        raise ValueError("The 'log_popula' column is missing in the dataset.")
    # return gdf
    y = gdf['log_popula'].to_numpy(dtype=np.float32)

    return torch.tensor(coords), torch.tensor(y)


def visualize_pop(coords, y, cont):
    plt.figure()
    sc = plt.scatter(coords[:,0], coords[:,1], c=y, cmap='viridis', s=1)
    plt.colorbar(sc, label='Log Population')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(cont + ' Population Density')
    plt.grid(False)
    plt.show()


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
    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_model.fit(x_train, y_train)
    test_predictions = rf_model.predict(x_test)
    return rf_model, test_predictions

def train_knn(x_train, y_train, x_test, n_neighbors=5):
    knn_model = KNeighborsRegressor(n_neighbors=n_neighbors, metric='euclidean')
    knn_model.fit(x_train, y_train)
    test_predictions = knn_model.predict(x_test)
    return knn_model, test_predictions


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

def train_mlp( x_train_sub, y_train_sub, x_test, y_test):
    pred_model = MLP(input_dim=256, dim_hidden=64, num_layers=2, out_dims=1).float().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(pred_model.parameters(), lr=0.001)

    losses = []
    epochs = 3000

    for epoch in range(epochs):
        optimizer.zero_grad()
        # Forward pass
        y_pred = pred_model(x_train_sub.float().to(device))
        # Compute the loss
        loss = criterion(y_pred.reshape(-1), y_train_sub.float().to(device))
        # Backward pass
        loss.backward()
        # Update the parameters
        optimizer.step()
        # Append the loss to the list
        losses.append(loss.item())
        # if (epoch + 1) % 250 == 0:
        #   print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")

    with torch.no_grad():
        pred_model.eval()
        y_pred_test = pred_model(x_test.float().to(device))

    # mse = criterion(y_pred_test.reshape(-1), y_test.float().to(device)).item()

    return pred_model, y_pred_test

def evaluate_regression(y_test, predictions):

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return mse, mae, r2


all_results = {}
samples = {}
results_csv = {
    "Pre-training model": [],
    "Continent": [],
    "Number of Finetuning samples": [],
    "MSE": [],
    "MAE":[],
    "R2":[],
    "Finetuning model type": []
}


samples_range = [10, 20, 30, 40, 50, 100, 200, 400, 800, 1600, 3200, 6500, 13000]

for continent in continent_names:
    for model_name in models.keys():
        print(f"Running Downstream for {continent}, with {model_name} pretraining")

        coords, y = get_population_data(continent_name=continent)
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

                    rf_model, test_predictions = train_random_forest(x_train_samples_np, y_train_samples_np, x_test_np)

                    # Calculate metrics
                    test_predictions_np = test_predictions
                    mse, mae, r2 = evaluate_regression(y_test_np, test_predictions_np)
                    
                if downstream_model_type == "KNN":
                    x_train_samples_np = x_train_samples.cpu().numpy()
                    y_train_samples_np = y_train_samples.cpu().numpy()
                    x_test_np = x_test.cpu().numpy()
                    y_test_np = y_test.cpu().numpy()

                    knn_model, test_predictions = train_knn(x_train_samples_np, y_train_samples_np, x_test_np)

                    # Calculate metrics
                    test_predictions_np = test_predictions
                    mse, mae, r2 = evaluate_regression(y_test_np, test_predictions_np)

                if downstream_model_type == "MLP":
                    mlp_model, test_predictions = train_mlp( x_train_samples, y_train_samples, x_test, y_test)

                    y_test_np = y_test.cpu().numpy()
                    test_predictions_np = test_predictions.cpu().numpy()
                    mse, mae, r2 = evaluate_regression(y_test_np, test_predictions_np)

                print(f'{downstream_model_type} - {continent} with {n_samples} samples:')
                print(f'  Test Error (MSE): {mse}, MAE: {mae}, R2: {r2}')

                # Write results to CSV format
                results_csv["Pre-training model"].append(model_name)
                results_csv["Continent"].append(continent)
                results_csv["Number of Finetuning samples"].append(n_samples)
                results_csv["MSE"].append(mse)
                results_csv["MAE"].append(mae)
                results_csv["R2"].append(r2)
                results_csv["Finetuning model type"].append(downstream_model_type)
    
            if n_samples >= max_samples:
                samples_range.remove(max_samples)
                break
            
    df = pd.DataFrame(results_csv)
    df.to_csv(f'_population_Results_{downstream_model_type}.csv', index=False)

df = pd.DataFrame(results_csv)
df.to_csv(f'population_Results_{downstream_model_type}.csv', index=False)
