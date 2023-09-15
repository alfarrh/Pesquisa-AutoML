import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import time
import sys
import pickle
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

from sklearn.base import clone
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from pandas import DataFrame

global rs, folds, test_split, exec_time

rs = 1              # Random State
folds = 10          # Quantidade de Folds
test_split = 0.30   # Test split para os modelos
exec_time = 60     # Quantidade de tempo para rodar cada AutoML

def print_string(features_list):
    '''
    Função para printar as features de forma reduzida
    '''
    
    features_string = '['
    for i in range(10): #Se a quantidade de features > 10 adicionar um ... e parar de escrever
        features_string += f"{features_list[i]}"
        if i == 9:
            features_string += ', ...]'
        else:
            features_string += ', '
    
    return features_string

def compute_shap(model, feature_names, X_train, explainer_ = 'kernel', num_samples = 50) -> DataFrame:
    '''
    Função principal do shap, aqui serão computadas as features importances de cada atributo
    model: Modelo fitado em algum dataset
    feature_names: Nome das features, importante para deixar o gráfico mais legível
    X_train: Conjunto de dados no qual o shap irá se basear para gerar as importances
    explainer_: Qual explainer do shap será utilizado, no momento atual somente o parâmetro 'tree' é suportado
    já o parâmetro 'kernel' foi colocado como placeholder
    num_samples: Quantidade de amostras que serão selecionadas do conjunto X_train, esse parâmetro é utilizado para
    reduzir o tempo de processamento do shap, já que dependendo do tamanho da base isso pode demorar várias horas sem necessidade    
    '''

    shap.initjs() #Inicia ambiente javascript, utilizado para criar gráficos

    #Escolhe as amostras de forma aleatória
    sample = shap.sample(X_train, num_samples, random_state=rs)
    #Define o explainer entre tree e kernel (Padrão: kernel)
    explainer = shap.TreeExplainer(model) if explainer_ == 'tree' else shap.KernelExplainer(model.predict, sample)
    #Inicia calculo de contribuição de cada feature (shapley values)
    #Devolve um np.array contendo valores positivos e negativos
    shap_values = explainer.shap_values(sample)
    
    #Pego o valor
    shap_sum = np.abs(shap_values).mean(axis=0)

    #Make a dataframe of each feature and shap_value
    importance_df = pd.DataFrame([feature_names, shap_sum.tolist()]).T
    importance_df.columns = ['feature', 'importance']
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    features = importance_df['feature'].index.to_list()
    
    #shap.summary_plot(shap_values, sample, feature_names=feature_names)
    
    return features

def feature_selection(model_, ordered_features, X_train, X_test, y_train, y_test, n_min = -1, condition = 'features', threshold=0.03):
    
    features_list = ordered_features[:n_min].copy() if n_min >= 0 else ordered_features.copy()
    
    best_features = None
    first_acc = 0
    
    first_it = True
    current = features_list.copy()
    
    params = model_.get_params()

    for feature in features_list[1:][::-1]:

        current.remove(feature)
        
        X_train_shap = X_train[:, current]
        X_test_shap = X_test[:, current]
        
        if 'max_features' in params:
            if len(current) == params['max_features']:
                print('Limite do modelo.')

        if 'warm_start' in params:
            model2 = clone(model_)
        else:
            model2 = model_
        
        model2.fit(X_train_shap, y_train)
        y_pred = model2.predict(X_test_shap)
        test_acc = accuracy_score(y_test, y_pred)
        
        if first_it:
            first_it = False
            first_acc = test_acc
        
        if condition == 'accuracy':
            if test_acc > first_acc:
                first_acc = test_acc
        
        if test_acc > first_acc-first_acc*threshold:
            current_acc = test_acc
            best_features = current.copy()
            print(f'Best accuracy: {current_acc} with {best_features} features')
    
    if len(best_features) >= 10:
        total = print_string(best_features)
        print(f'Results: Best accuracy: {current_acc} with {total} total of {len(best_features)} features')
    else: print(f'Results: Best accuracy: {current_acc} with {best_features}')
    
    return best_features

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def run_test(models: list, X_train_, y_train_ ,X_test_ , y_test_, cm = False, new_features = None):
    
    if new_features != None:
        X_train_ = X_train_[:,new_features]
        X_test_ = X_test_[:,new_features]
    
    for model in models:

        model_ = clone(model)
        
        model_.fit(X_train_, y_train_)
        y_pred = model_.predict(X_test_)
        test_acc = accuracy_score(y_test_, y_pred)

        print(model_)
        print("Accuracy: ", test_acc, end="\n")
        if new_features != None:
            print("Features: ", new_features, end="\n")
    
        if cm:
            matrix = ConfusionMatrixDisplay(confusion_matrix(y_test_, y_pred))
            matrix.plot()
            
def SVM_Model(X_train, y_train):
    
    #Tunning
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    
    param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.1, 1],
    'coef0': [0.0, 0.5, 1.0],
    'probability': [True],
    'random_state': [rs]
    }
    
    model = SVC()
    model.fit(X_train, y_train)
    kfold = StratifiedKFold(n_splits= folds, random_state= rs, shuffle=True)
    
    print("Grid Search...")
    grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv= kfold)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

def RunAutoSklearn(X_train, y_train):
    
    from autosklearn.classification import AutoSklearnClassifier
    from autosklearn.metrics import balanced_accuracy, precision, recall, f1

    minutos = exec_time

    autosk = AutoSklearnClassifier(
        include = {
            "classifier": ["random_forest", "decision_tree", "extra_trees",
                        'liblinear_svc', 'libsvm_svc','k_nearest_neighbors'],
            "feature_preprocessor":["no_preprocessing"]
        },
        time_left_for_this_task= minutos*60,
        per_run_time_limit= 30,
        scoring_functions= [balanced_accuracy, precision, recall, f1],
        ensemble_class= 'none',
        #ensemble_nbest = 25,
        n_jobs = -1
    )

    autosk.fit(X_train, y_train, dataset_name= 'Test Base')

    models = autosk.show_models()
    best_model = list(models.keys())[0]
    sklearn_model = models[best_model]["sklearn_classifier"]

    sklearn_classifier = clone(sklearn_model)
    
    return sklearn_classifier, autosk
    
def RunFlaML(X_train, y_train):
    
    from flaml import AutoML

    minutos = exec_time

    fla_automl = AutoML()

    fla_automl_settings = {
        "time_budget": minutos*60,
        "metric": 'accuracy',
        "n_jobs": -1,
        "ensemble": False,
        "n_splits": folds
    }

    fla_automl.fit(X_train, y_train, task= "classification", **fla_automl_settings)
    
    flaml_classifier = clone(fla_automl.model.estimator)
    
    return flaml_classifier

def VennDiagram(svm_features, autosk_features, flaml_features, nome):
    
    from matplotlib_venn import venn3_unweighted

    set1 = set(svm_features)
    set2 = set(autosk_features)
    set3 = set(flaml_features)

    subsets = (set1 - set2 - set3,
            set2 - set1 - set3,
            set1 & set2 - set3,
            set3 - set1 - set2,
            set1 & set3 - set2,
            set2 & set3 - set1,
            set1 & set2 & set3)

    venn3 = venn3_unweighted(subsets, set_labels=('SVM', 'AutoSklearn', 'FlaML'))

    print('Diagrama de Venn')
    plt.savefig(f'results/Venn_{nome}.png')

def AutoML(base, nome):

    print('Running automl process for: ', exec_time*2)
    
    with open(base, 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    y = data['y']
    feature_names = data['feature_names']
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= test_split, random_state=rs)
    
    start_time = time.time()
    
    print('Tunning SVM...')
    svm_model = SVM_Model(X_train, y_train)
    svm_model.fit(X_train, y_train)
    
    print('Computing Shap (SVM)...)')
    svm_features = compute_shap(svm_model, feature_names, X_train)

    print('Feature Selection (SVM)...')
    svm_best_features = feature_selection(svm_model, svm_features, X_train, X_test, y_train, y_test)
    
    svm_timer = time.time() - start_time
    start_time = time.time()
    
    print('AutoSKLearn...')
    autosk_model, autosk = RunAutoSklearn(X_train, y_train)
    autosk_model.fit(X_train, y_train)
    
    print('Computing Shap (AutoSKLearn)...')
    autosk_features = compute_shap(autosk_model, feature_names, X_train)
    
    print('Feature Selection (AutoSKLearn)...')
    autosk_best_features = feature_selection(autosk_model, autosk_features, X_train, X_test, y_train, y_test)
    
    autosk_timer = time.time() - start_time
    start_time = time.time()
    
    print('FlaML...')
    flaml = RunFlaML(X_train, y_train)
    flaml.fit(X_train, y_train)
    
    print('Computing Shap (FlaML)...')
    flaml_features = compute_shap(flaml, feature_names, X_train)
    
    print('Feature Selection (FlaML)...')
    flaml_best_features = feature_selection(flaml, flaml_features, X_train, X_test, y_train, y_test)
    
    flaml_timer = time.time() - start_time

    orig = sys.stdout
    filename = "results/resultados_" + nome +".txt"
    with open(filename, 'w') as f:

        sys.stdout = f
    
        print('Running tests (SVM)...')
        run_test([svm_model], X_train, y_train, X_test, y_test, new_features=svm_best_features)
        print('Timer: ', svm_timer/60, end='\n\n')
        print('Running tests (AutoSKLearn)...')
        run_test([autosk_model], X_train, y_train, X_test, y_test, new_features=autosk_best_features)
        print('Timer: ', autosk_timer/60, end='\n\n')
        autosk.sprint_statistics()
        print('Running tests (FlaML)...')
        run_test([flaml], X_train, y_train, X_test, y_test, new_features=flaml_best_features)
        print('Timer: ', flaml_timer/60, end='\n\n')
    
    sys.stdout = orig

    VennDiagram(svm_best_features, autosk_best_features, flaml_best_features, nome)


if __name__ == "__main__":
    
    c, arg1, arg2 = sys.argv
    print('Base escolhida: ', arg2)
    AutoML(arg1, arg2)
