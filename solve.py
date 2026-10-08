import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_validate, KFold
import warnings

warnings.filterwarnings('ignore')

def process_variant(train_file, test_file, variant_name, degrees):
    print(f"\nProcessing {variant_name}...")
    
    # Load data
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    X_train = train_df.drop(columns=['y'])
    y_train = train_df['y']
    X_test = test_df
    
    alphas = [0.01, 0.1, 1.0, 10.0]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    
    for degree in degrees:
        print(f"  Evaluating degree {degree}...")
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        scaler = StandardScaler()
        
        models = [
            ('OLS', LinearRegression(), None)
        ]
        
        for alpha in alphas:
            models.append(('Ridge', Ridge(alpha=alpha), alpha))
            models.append(('Lasso', Lasso(alpha=alpha, max_iter=100000), alpha))
            models.append(('ElasticNet', ElasticNet(alpha=alpha, l1_ratio=0.5), alpha))
            
        for method, estimator, alpha in models:
            print(f"    Running {method} with alpha={alpha}...")
            pipeline = Pipeline([
                ('poly', poly),
                ('scaler', scaler),
                ('model', estimator)
            ])
            
            cv_results = cross_validate(
                pipeline, X_train, y_train, cv=kf, 
                scoring=('r2', 'neg_mean_squared_error'),
                return_train_score=False,
                n_jobs=None
            )
            
            mean_r2 = np.mean(cv_results['test_r2'])
            mean_mse = -np.mean(cv_results['test_neg_mean_squared_error'])
            
            results.append({
                'Variant': variant_name,
                'Method': method,
                'Degree': degree,
                'Alpha': alpha if alpha is not None else 'N/A',
                'CV_R2': mean_r2,
                'CV_MSE': mean_mse
            })
            
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'model_comparison_{variant_name}.csv', index=False)
    
    # Find best model based on CV_R2
    best_idx = results_df['CV_R2'].idxmax()
    best_model = results_df.loc[best_idx]
    
    print(f"\nBest Model for {variant_name}:")
    print(best_model)
    
    # Refit best model on full training data
    best_degree = best_model['Degree']
    best_method = best_model['Method']
    best_alpha = best_model['Alpha']
    
    poly = PolynomialFeatures(degree=best_degree, include_bias=False)
    scaler = StandardScaler()
    
    if best_method == 'OLS':
        estimator = LinearRegression()
    elif best_method == 'Ridge':
        estimator = Ridge(alpha=best_alpha)
    elif best_method == 'Lasso':
        estimator = Lasso(alpha=best_alpha, max_iter=100000)
    elif best_method == 'ElasticNet':
        estimator = ElasticNet(alpha=best_alpha, l1_ratio=0.5)
        
    pipeline = Pipeline([
        ('poly', poly),
        ('scaler', scaler),
        ('model', estimator)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = pipeline.predict(X_test)
    pred_df = pd.DataFrame({'y': y_pred})
    pred_df.to_csv(f'IMT2024086_pred_{variant_name}.csv', index=False)
    
    # Generate Plots
    generate_plots(results_df, variant_name, best_degree)
    
    return best_model

def generate_plots(results_df, variant_name, best_degree):
    # We will plot the best score for each degree
    # Group by degree and get max R2 and min MSE
    best_per_degree = results_df.groupby('Degree').agg({
        'CV_R2': 'max',
        'CV_MSE': 'min'
    }).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Degree vs CV MSE
    ax1.plot(best_per_degree['Degree'], best_per_degree['CV_MSE'], marker='o')
    ax1.axvline(x=best_degree, color='r', linestyle='--', label=f'Best Degree: {best_degree}')
    ax1.set_xlabel('Polynomial Degree')
    ax1.set_ylabel('Mean CV MSE')
    ax1.set_title(f'{variant_name}: Degree vs CV MSE')
    ax1.legend()
    ax1.grid(True)
    
    # Degree vs CV R2
    ax2.plot(best_per_degree['Degree'], best_per_degree['CV_R2'], marker='o', color='green')
    ax2.axvline(x=best_degree, color='r', linestyle='--', label=f'Best Degree: {best_degree}')
    ax2.set_xlabel('Polynomial Degree')
    ax2.set_ylabel('Mean CV R^2')
    ax2.set_title(f'{variant_name}: Degree vs CV R^2')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{variant_name}_plots.png')
    plt.close()

if __name__ == "__main__":
    best_var1 = process_variant('IMT2024086_train_var1.csv', 'IMT2024086_test_var1.csv', 'var1', range(1, 11))
    best_var2 = process_variant('IMT2024086_train_var2.csv', 'IMT2024086_test_var2.csv', 'var2', range(1, 21))
    print("\nProcessing complete. Predictions and plots generated.")
