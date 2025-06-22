# Music Recommendation System Overview

This project implements a music recommendation system utilizing content-based and item-based collaborative filtering approaches. It encompasses preprocessing, hyperparameter tuning, experiments, and statistical significance testing. The system leverages acoustic features and user interactions from 2012, targeting users aged 12–18, with a default configuration for age 16.

## Workflow

### Preprocessing

Follow these steps to prepare the dataset:

1. **Filter Data by Year and User Age**  
   Run `year_and_users_age_filtering.py` to extract interactions from 2012 for users aged 12–18 with available acoustic features.

2. **Data Processing**  
   Run `data_processing.py` to:  
   - Apply user and item-based k-core filtering.  
   - Balance user groups.  
   - Create training, validation, and test splits.

3. **Final Dataset Balancing**  
   Run `final_dataset_balancing.py` to ensure balanced datasets.

4. **Extract Track Features**  
   Run `extract_track_features.py` to retrieve acoustic features for tracks in the training, validation, and test sets.

5. **Split Track Features**  
   Run `split_track_features.py` to split the extracted features into appropriate subsets.

6. **Handle Missing Values (Optional)**  
   If undefined values are found in track features due to random sampling, run `drop_songs_na_features.py` to remove problematic songs.

### Hyperparameter Tuning

Run `tuning_hyper_parameters.py` to optimize hyperparameters for the recommendation model. The best hyperparameters (for age 16) are:

- **Baseline**:  
  ```python
  {
      'maxk': 10,
      'shrink': 0,
      'similarity': 'cosine',
      'normalize': True,
      'asymmetric_alpha': False,
      'tversky_alpha': False,
      'tversky_beta': False
  }
  ```

- **Features**:  
  All feature hyperparameters are identical to the baseline except for `mode`, which uses:  
  ```python
  {
      'maxk': 10,
      'shrink': 10,
      'similarity': 'cosine',
      'normalize': True,
      ...
  }
  ```

### Running Experiments

Run `recommender_experiments.py` to execute the recommendation experiments. Configuration options include:

- **Content-Based vs. Item-KNN**:  
  For content-based recommendations, use the `similarity` variable (line 143). For pure item-KNN, switch to `similarity2`.

- **Age Group**:  
  Default experiments are configured for age 16. To change the age group, update the `test_path` in the script.

### Statistical Significance Testing

Run `statistical_significance.py` to perform statistical tests on experiment results.

### Summary and Analysis

- **Summary Data**:  
  Run `summary_data.py` to generate dataset statistics.

- **Genre Distribution**:  
  Run `genre_count.py` to compute genre counts.

## Notes

- **Directory Paths**: Ensure all input and output paths are updated to match your local environment.
- **Age Group Customization**: The default setup targets age 16. Modify `test_path` in `recommender_experiments.py` for other age groups within 12–18.
- **Missing Values**: The need to run `drop_songs_na_features.py` depends on random sampling outcomes. Check for undefined values before running experiments.
- **Hyperparameters**: The provided hyperparameters are optimized for the baseline and acoustic features. Adjust as needed for other configurations.

## Usage Example

### Preprocessing
```bash
python year_and_users_age_filtering.py
python data_processing.py
python final_dataset_balancing.py
python extract_track_features.py
python split_track_features.py
python drop_songs_na_features.py  # Optional
```

### Hyperparameter Tuning
```bash
python tuning_hyper_parameters.py
```

### Experiments
```bash
python recommender_experiments.py
```

### Statistical Tests
```bash
python statistical_significance.py
```

### Summaries
```bash
python summary_data.py
python genre_count.py
```