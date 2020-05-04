import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import stats
import math
import copy
import dill
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier


def set_of_tests_for_classifier(n_classes, model_type, alpha, decalibration):
    distances = ["abs", "sq", "log"]
    ece_types = ["cwes", "cwew", "cfes", "cfew"]
    cw_cf = {"cwes": True, "cwew": True, "cfes": False, "cfew": False}
    ew_es = {"cwes": False, "cwew": True, "cfes": False, "cfew": True}

    d = dict()
    for ece_type in ece_types:
        d[ece_type] = dict()

    for distance in distances:
        ECE_calc = distance
        for ece_type in ece_types:
            model = TestModel(random_seed=0, n_classes=n_classes, model_type=model_type, alpha=alpha, ECE_calc=ECE_calc,
                              decalibration=decalibration)
            data = model.tests_for_thesis(cw_cf[ece_type], ew_es[ece_type])
            d[ece_type][distance] = data

    return d


class TestModel:

    def __init__(self, random_seed, n_classes, model_type, alpha, ECE_calc, decalibration, n_data_points=100):
        self.random_seed = random_seed
        self.__reset_random_seed()
        self.n_classes = n_classes
        self.model_type = model_type
        self.alpha = alpha
        self.ECE_calc = ECE_calc
        self.decalibration = decalibration
        if model_type == "forest_cover_decision_tree":
            self.__train_forest_cover_decision_tree()
        elif model_type == "forest_cover_random_forest":
            with open('../forest_cover_random_forest_predictions.pkl', 'rb') as file:
                self.predictions = dill.load(file)
            with open('../forest_cover_random_forest_labels.pkl', 'rb') as file:
                self.labels = dill.load(file)
            self.pred_index = 0

        self.bin_placement = "random"
        self.n_data_points = n_data_points
        self.resampling_tests_per_value = 100
        self.p_values_per_distribution = 1000
        self.equal_width_bins = [5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 175, 200, 250, 500]
        self.equal_size_bins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

    def __reset_random_seed(self):
        np.random.seed(self.random_seed)

    def __reset_pred_index(self):
        self.pred_index = 0

    def __train_forest_cover_decision_tree(self):
        # Model inspiration https://www.kaggle.com/jiashenliu/random-forest-with-feature-selection-0-95-accuracy

        df = pd.read_csv('covtype.csv')
        train, test = train_test_split(df, test_size=0.2, random_state=999)

        train_Y = train['Cover_Type']
        test_Y = test['Cover_Type']

        # feature selection
        train_X = train.iloc[:, 0:54]
        clf = ExtraTreesClassifier()
        clf = clf.fit(train_X, train_Y)
        model = SelectFromModel(clf, prefit=True)

        train_X = model.transform(train.iloc[:, 0:54])
        test_X = model.transform(test.iloc[:, 0:54])

        clf = DecisionTreeClassifier(min_samples_leaf=self.alpha, random_state=999)
        clf = clf.fit(train_X, train_Y)
        self.predictions = clf.predict_proba(test_X)
        self.labels = np.asarray(test_Y) - 1
        self.pred_index = 0

    def __p_value(self, at_value, values, len_values):
        return sum(np.asarray(values) >= at_value) / len_values

    def __class_J_ECE(self, class_J_predictions, is_J_class, n_bins, equal_width_bins,
                      class_name="0"):

        data = np.array((class_J_predictions, is_J_class)).T

        # Dividing datapoints into bins

        if equal_width_bins:
            bins = [[] for i in range(n_bins)]
            bin_width = 1.0 / n_bins

            for d in data:
                idx = min(int(d[0] // bin_width), n_bins - 1)
                bins[idx].append(d)

        else:
            sorted_data = data[data[:, 0].argsort()]
            if self.bin_placement == "random":
                random_bins = np.array_split(np.random.rand(self.n_data_points), n_bins)
                np.random.shuffle(random_bins)
                idx = 0
                for i in range(len(random_bins)):
                    random_bins[i] = np.ndarray.tolist(random_bins[i])
                    for j in range(len(random_bins[i])):
                        random_bins[i][j] = sorted_data[idx]
                        idx += 1
                    random_bins[i] = np.asarray(random_bins[i])
                bins = random_bins
            else:
                bins = np.array_split(sorted_data, n_bins)

        # Calculating ECE

        ECE_score = 0
        for b in bins:
            b = np.asarray(b)
            bin_size = len(b)
            if bin_size == 0:
                continue
            average_predictions_in_bin = b[:, 0].mean()
            actual_proportion_in_bin = b[:, 1].mean()

            if self.ECE_calc == "absolute":
                ECE_score += abs(actual_proportion_in_bin - average_predictions_in_bin) * bin_size / self.n_data_points
            elif self.ECE_calc == "square":
                ECE_score += (
                                     actual_proportion_in_bin - average_predictions_in_bin) ** 2 * bin_size / self.n_data_points
            elif self.ECE_calc == "log":
                if actual_proportion_in_bin == average_predictions_in_bin:
                    ECE_score += 0
                elif actual_proportion_in_bin < 1 and average_predictions_in_bin == 1:
                    ECE_score += 99999
                elif actual_proportion_in_bin > 0 and average_predictions_in_bin == 0:
                    ECE_score += 99999
                else:
                    if actual_proportion_in_bin == 0:
                        first_log = 0
                    else:
                        first_log = actual_proportion_in_bin * math.log(
                            actual_proportion_in_bin / average_predictions_in_bin)
                    if actual_proportion_in_bin == 1:
                        second_log = 0
                    else:
                        second_log = (1.0 - actual_proportion_in_bin) * math.log(
                            (1.0 - actual_proportion_in_bin) / (1.0 - average_predictions_in_bin))
                    ECE_score += (first_log + second_log) * bin_size / self.n_data_points


        return ECE_score

    def __confidence_ECE(self, predictions, labels, n_bins, equal_width_bins):

        max_preds = np.amax(predictions, axis=1)
        max_pred_idx = np.argmax(predictions, axis=1)
        is_max_class = labels == max_pred_idx

        return self.__class_J_ECE(max_preds,
                                  is_max_class,
                                  n_bins,
                                  equal_width_bins)

    def __classwise_ECE(self, predictions, labels, n_bins, equal_width_bins):

        ECE_score = 0

        for class_label in range(0, self.n_classes, 1):
            class_J_predictions = predictions[:, class_label]
            is_J_class = labels == class_label

            ECE_score += self.__class_J_ECE(class_J_predictions,
                                            is_J_class,
                                            n_bins,
                                            equal_width_bins,
                                            class_name="Class " + str(class_label)) / self.n_classes

        return ECE_score

    def __predictions(self):
        if self.model_type == "dirichlet01":
            return np.random.dirichlet([0.1] * self.n_classes, self.n_data_points)
        elif self.model_type == "dirichlet02":
            return np.random.dirichlet([0.2 - 0.02 * i for i in range(0, self.n_classes)], self.n_data_points)
        elif self.model_type == "forest_cover_decision_tree" or "forest_cover_random_forest":
            preds = self.predictions[self.pred_index: self.pred_index + self.n_data_points]
            self.pred_index += self.n_data_points
            return preds
        else:
            raise ValueError('ModelType not defined')

    def __decalibrate_method2(self, predictions):
        pred = copy.deepcopy(predictions)

        for i in range(self.n_data_points):
            max_pred = np.max(pred[i])
            if max_pred == 1.0:
                continue
            for j in range(len(pred[i])):
                if pred[i][j] == max_pred:
                    if max_pred + self.alpha > 0:
                        pred[i][j] += self.alpha

            pred[i] = pred[i] / np.sum(pred[i])

        return pred

    def resampling_test(self,
                        times,
                        n_bins,
                        equal_width_bins,
                        use_classwise_ECE):

        # Model predictions

        predictions = self.__predictions()

        # Real labels

        if self.model_type == "forest_cover_decision_tree" or self.model_type == "forest_cover_random_forest":
            labels = self.labels[self.pred_index - self.n_data_points: self.pred_index]

        elif self.decalibration == "method1":
            labels_cal = np.array([np.random.choice(range(0, self.n_classes, 1), p=preds) for preds in predictions])
            labels_miscal = np.array(
                [np.random.choice(range(0, self.n_classes, 1), p=[1.0 / self.n_classes] * self.n_classes) for i in
                 range(self.n_data_points)])

            break_point = int(self.n_data_points * self.alpha)
            labels = np.append(labels_cal[:break_point], labels_miscal[break_point:])

        elif self.decalibration == "method2":
            real_dist = self.__decalibrate_method2(predictions)
            real_dist, predictions = predictions, real_dist
            labels = np.array([np.random.choice(range(0, self.n_classes, 1), p=preds) for preds in real_dist])

        # Real ECE

        if use_classwise_ECE:
            score = self.__classwise_ECE(predictions, labels, n_bins, equal_width_bins)
        else:
            score = self.__confidence_ECE(predictions, labels, n_bins, equal_width_bins)

        # Perfect ECE distribution from consistency resampling

        scores = []

        for i in range(times):
            labels = np.array([np.random.choice(range(0, self.n_classes, 1), p=preds) for preds in predictions])

            if use_classwise_ECE:
                scores.append(self.__classwise_ECE(predictions, labels, n_bins, equal_width_bins))
            else:
                scores.append(self.__confidence_ECE(predictions, labels, n_bins, equal_width_bins))

        return (score, scores)

    def resampling_test_p_value_distribution(self,
                                             values_per_distribution,
                                             resampling_tests_per_value,
                                             n_bins,
                                             equal_width_bins,
                                             use_classwise_ECE):
        self.__reset_random_seed()
        self.__reset_pred_index()

        p_values = []
        for i in range(values_per_distribution):
            results = self.resampling_test(resampling_tests_per_value, n_bins,
                                           equal_width_bins,
                                           use_classwise_ECE)
            p_values.append(self.__p_value(results[0], results[1], resampling_tests_per_value))
        return p_values

    def __describe(self, results, titles):
        df = pd.DataFrame(columns=["bins", "ece", "binning", "model", "n_classes",
                                   "quantile at 0.01", "quantile at 0.05",
                                   "quantile at 0.10", "quantile at 0.90", "quantile at 0.95", "mean", "variance",
                                   "values_in_distribution", "resampling_tests_per_p", "n_data_points"])

        for i in range(len(results)):
            r = results[i]
            s = stats.describe(r)

            df = df.append({
                "bins": titles[i][0],
                "ece": titles[i][1],
                "binning": titles[i][2],
                "model": titles[i][3],
                "n_classes": titles[i][4],
                "quantile at 0.01": np.round(1 - self.__p_value(0.01, np.asarray(r), len(r)), 3),
                "quantile at 0.05": np.round(1 - self.__p_value(0.05, np.asarray(r), len(r)), 3),
                "quantile at 0.10": np.round(1 - self.__p_value(0.10, np.asarray(r), len(r)), 3),
                "quantile at 0.90": np.round(1 - self.__p_value(0.90, np.asarray(r), len(r)), 3),
                "quantile at 0.95": np.round(1 - self.__p_value(0.95, np.asarray(r), len(r)), 3),
                "mean": np.round(s.mean, 4),
                "variance": np.round(s.variance, 4),
                "values_in_distribution": titles[i][5],
                "resampling_tests_per_p": titles[i][6],
                "n_data_points": titles[i][7]
            }, ignore_index=True)

        return df

    def tests_for_thesis(self, use_classwise_ECE, use_equal_width_bins):
        if use_equal_width_bins:
            bins = self.equal_width_bins
            binning = "ew"
        else:
            bins = self.equal_size_bins
            binning = "es"

        if use_classwise_ECE:
            ece = "cw"
        else:
            ece = "cf"

        results = []
        for n_bins in bins:
            results.append(self.resampling_test_p_value_distribution(self.p_values_per_distribution,
                                                                     self.resampling_tests_per_value,
                                                                     n_bins=n_bins,
                                                                     equal_width_bins=use_equal_width_bins,
                                                                     use_classwise_ECE=use_classwise_ECE))

        return self.__describe(results, [
            [n_bins, ece, binning, self.model_type + "_" + str(self.alpha), self.n_classes,
             self.p_values_per_distribution,
             self.resampling_tests_per_value, self.n_data_points] for n_bins in bins])
