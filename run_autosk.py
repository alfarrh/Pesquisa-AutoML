from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import time
import sys

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
        n_jobs = 4
    )

    autosk.fit(X_train, y_train, dataset_name= 'Test Base')

    models = autosk.show_models()
    best_model = list(models.keys())[0]

    sklearn_classifier = clone(models[best_model]["sklearn_classifier"])
    
    print(sklearn_classifier, '\n', autosk.sprint_statistics())
    
    return sklearn_classifier

def run_test(models: list, X_train_, y_train_ ,X_test_ , y_test_, cm = False, new_features = None):
    
    if new_features != None:
        print("New features: ", new_features, end="\n\n")
        X_train_ = X_train_[:,new_features]
        X_test_ = X_test_[:,new_features]
    
    for model in models:

        model_ = clone(model)
        
        model_.fit(X_train_, y_train_)
        y_pred = model_.predict(X_test_)
        test_acc = accuracy_score(y_test_, y_pred)

        print(model_)
        print("Accuracy: ", test_acc, end="\n\n")
    
        if cm:
            matrix = ConfusionMatrixDisplay(confusion_matrix(y_test_, y_pred))
            matrix.plot()
            
            
def AutoML(base, nome):
    
    with open(base, 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    y = data['y']
    feature_names = data['feature_names']
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= test_split, random_state=rs)
    
    start_time = time.time()
    
    print('AutoSKLearn...')
    autosk_model = RunAutoSklearn(X_train, y_train)
    autosk_model.fit(X_train, y_train)
    
    autosk_timer = time.time() - start_time
    print('Timer:', autosk_timer)
   
    print('Running tests (AutoSKLearn)...')
    run_test([autosk_model], X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    
    c, arg1, arg2 = sys.argv
    
    global exec_time, test_split, rs
    
    exec_time = int(arg2)
    test_split = 0.3
    rs = 1
    
    print('Base escolhida: ', arg1)
    AutoML(arg1, arg1)
