from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime
import json
import numpy

from app import app

db = SQLAlchemy(app)
migrate = Migrate(app, db)

def default(o):
    if isinstance(o, numpy.int64): return int(o)
    if isinstance(o, numpy.int32): return int(o)
    if isinstance(o, numpy.float64): return float(o)
    if isinstance(o, numpy.float32): return float(o)
    raise TypeError

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String)
    split_uuid = db.Column(db.String)
    created_on = db.Column(db.DateTime, default=datetime.utcnow)

    model = db.Column(db.String)
    train_data_stats = db.Column(db.String)
    validation_data_stats = db.Column(db.String)
    test_data_stats = db.Column(db.String)

    train_accuracy = db.Column(db.Float)
    train_loss = db.Column(db.Float)
    accuracy = db.Column(db.Float)
    loss = db.Column(db.Float)
    test_accuracy = db.Column(db.Float)
    test_loss = db.Column(db.Float)

    probabilities = db.Column(db.String)
    labels = db.Column(db.String)
    test_probabilities = db.Column(db.String)
    test_labels = db.Column(db.String)

    description = db.Column(db.String)
    input_form = db.Column(db.String)
    label = db.Column(db.String)

    hyperparameters = db.Column(db.String)

    history = db.Column(db.String)

    def __repr__(self):
        return '<Result accuracy: {}>'.format(self.accuracy)

    def __init__(self,
            model,
            uuid,
            split_uuid,
            train_data_stats,
            validation_data_stats,
            test_data_stats,
            description,
            input_form,
            label,
            train_accuracy,
            train_loss,
            test_accuracy,
            test_loss,
            accuracy,
            loss,
            probabilities,
            labels,
            test_probabilities,
            test_labels,
            hyperparameters,
            history,
            ):
        self.model = model
        self.uuid = uuid
        self.split_uuid = split_uuid

        self.train_data_stats = json.dumps(train_data_stats, default=default)
        self.validation_data_stats = json.dumps(validation_data_stats, default=default)
        self.test_data_stats = json.dumps(test_data_stats, default=default)

        self.train_accuracy = train_accuracy
        self.train_loss = train_loss
        self.accuracy = accuracy
        self.loss = loss
        self.test_accuracy = test_accuracy
        self.test_loss = test_loss

        self.history = json.dumps(history, default=default)

        self.probabilities = json.dumps(probabilities, default=default)
        self.labels = json.dumps(labels, default=default)

        self.test_probabilities = json.dumps(test_probabilities, default=default)
        self.test_labels = json.dumps(test_labels, default=default)

        self.description = description
        self.input_form = input_form
        self.label = label

        self.hyperparameters = json.dumps(hyperparameters)

    def dict(self):
        return {
            "id": self.id,
            "uuid": self.uuid,
            "model": self.model,
            "createdOn": self.created_on.timestamp(),
            "trainDataStats": json.loads(self.train_data_stats),
            "validationDataStats": json.loads(self.validation_data_stats),
            "trainAccuracy": self.train_accuracy,
            "accuracy": self.accuracy,
            "input_form": self.input_form,
            # is this supposed to be label?
            "label": self.label,
        }

    def results(self):
        return json.loads(self.probabilites), json.loads(self.labels)

    def get_hyperparameters(self):
        return json.loads(self.hyperparameters)

    @property
    def split_seed(self):
        if self.split_uuid:
            return self.split_uuid
        return uuid

    @property
    def label_form(self):
        return self.label

class Expand_Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String)
    split_uuid = db.Column(db.String)
    created_on = db.Column(db.DateTime, default=datetime.utcnow)

    model = db.Column(db.String)
    train_data_stats = db.Column(db.String)
    validation_data_stats = db.Column(db.String)
    test_data_stats = db.Column(db.String)

    train_accuracy = db.Column(db.Float)
    train_loss = db.Column(db.Float)
    accuracy = db.Column(db.Float)
    loss = db.Column(db.Float)
    test_accuracy = db.Column(db.Float)
    test_loss = db.Column(db.Float)
    auc = db.Column(db.Float)
    sensitivity = db.Column(db.Float)
    specificity = db.Column(db.Float)
    npv = db.Column(db.Float)
    ppv = db.Column(db.Float)


    probabilities = db.Column(db.String)
    labels = db.Column(db.String)
    test_probabilities = db.Column(db.String)
    test_labels = db.Column(db.String)

    description = db.Column(db.String)
    input_form = db.Column(db.String)
    label = db.Column(db.String)

    hyperparameters = db.Column(db.String)

    history = db.Column(db.String)

    def __repr__(self):
        return '<Result accuracy: {}>'.format(self.accuracy)

    def __init__(self,
            model,
            uuid,
            split_uuid,
            train_data_stats,
            validation_data_stats,
            test_data_stats,
            description,
            input_form,
            label,
            train_accuracy,
            train_loss,
            test_accuracy,
            test_loss,
            auc,
            sensitivity,
            specificity,
            npv,
            ppv,
            accuracy,
            loss,
            probabilities,
            labels,
            test_probabilities,
            test_labels,
            hyperparameters,
            history,
            ):
        self.model = model
        self.uuid = uuid
        self.split_uuid = split_uuid

        self.train_data_stats = json.dumps(train_data_stats, default=default)
        self.validation_data_stats = json.dumps(validation_data_stats, default=default)
        self.test_data_stats = json.dumps(test_data_stats, default=default)

        self.train_accuracy = train_accuracy
        self.train_loss = train_loss
        self.accuracy = accuracy
        self.loss = loss
        self.test_accuracy = test_accuracy
        self.test_loss = test_loss
        self.auc = auc
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.npv = npv
        self.ppv = ppv
        self.history = json.dumps(history, default=default)

        self.probabilities = json.dumps(probabilities, default=default)
        self.labels = json.dumps(labels, default=default)

        self.test_probabilities = json.dumps(test_probabilities, default=default)
        self.test_labels = json.dumps(test_labels, default=default)

        self.description = description
        self.input_form = input_form
        self.label = label

        self.hyperparameters = json.dumps(hyperparameters)

    def dict(self):
        return {
            "id": self.id,
            "uuid": self.uuid,
            "model": self.model,
            "createdOn": self.created_on.timestamp(),
            "trainDataStats": json.loads(self.train_data_stats),
            "validationDataStats": json.loads(self.validation_data_stats),
            "trainAccuracy": self.train_accuracy,
            "accuracy": self.accuracy,
            "input_form": self.input_form,
            # is this supposed to be label?
            "label": self.label,
        }

    def results(self):
        return json.loads(self.probabilites), json.loads(self.labels)

    def get_hyperparameters(self):
        return json.loads(self.hyperparameters)

    @property
    def split_seed(self):
        if self.split_uuid:
            return self.split_uuid
        return uuid

    @property
    def label_form(self):
        return self.label

class CxResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    split = db.Column(db.String)
    run_id = db.Column(db.String)
    created_on = db.Column(db.DateTime, default=datetime.utcnow)

    fold = db.Column(db.Integer)
    total_folds = db.Column(db.Integer)
    model = db.Column(db.String)
    train_data_stats = db.Column(db.String)
    validation_data_stats = db.Column(db.String)
    test_data_stats = db.Column(db.String)
    holdout_test_data_stats = db.Column(db.String)
    train_accuracy = db.Column(db.Float)
    train_loss = db.Column(db.Float)
    accuracy = db.Column(db.Float)
    loss = db.Column(db.Float)
    test_accuracy = db.Column(db.Float)
    test_loss = db.Column(db.Float)
    auc = db.Column(db.Float)
    sensitivity = db.Column(db.Float)
    specificity = db.Column(db.Float)
    npv = db.Column(db.Float)
    ppv = db.Column(db.Float)
    holdout_test_accuracy = db.Column(db.Float)
    holdout_test_loss = db.Column(db.Float)

    probabilities = db.Column(db.String)
    labels = db.Column(db.String)
    test_probabilities = db.Column(db.String)
    test_labels = db.Column(db.String)
    holdout_test_probabilities = db.Column(db.String)
    holdout_test_labels = db.Column(db.String)

    description = db.Column(db.String)
    input_form = db.Column(db.String)
    label_form = db.Column(db.String)

    hyperparameters = db.Column(db.String)

    history = db.Column(db.String)

    test_f1_result = db.Column(db.Float)
    holdout_f1_result = db.Column(db.Float)


    def __repr__(self):
        return '<Result accuracy: {}>'.format(self.accuracy)

    def __init__(self,
                 fold,
                 total_folds,
                 model,
                 run_id,
                 split,
                 train_data_stats,
                 validation_data_stats,
                 test_data_stats,
                 holdout_test_data_stats,
                 description,
                 input_form,
                 label_form,
                 train_accuracy,
                 train_loss,
                 test_accuracy,
                 test_loss,
                 auc,
                 sensitivity,
                 specificity,
                 npv,
                 ppv,
                 accuracy,
                 loss,
                 holdout_test_accuracy,
                 holdout_test_loss,
                 probabilities,
                 labels,
                 test_probabilities,
                 test_labels,
                 holdout_test_probabilities,
                 holdout_test_labels,
                 hyperparameters,
                 history,
                 test_f1_result,
                 holdout_f1_result
                 ):

        self.fold = fold
        self.total_folds = total_folds
        self.model = model
        self.run_id = run_id
        self.split = split

        self.train_data_stats = json.dumps(train_data_stats, default=default)
        self.validation_data_stats = json.dumps(validation_data_stats, default=default)
        self.test_data_stats = json.dumps(test_data_stats, default=default)
        self.holdout_test_data_stats = json.dumps(holdout_test_data_stats, default=default)

        self.train_accuracy = train_accuracy
        self.train_loss = train_loss
        self.accuracy = accuracy
        self.loss = loss
        self.test_accuracy = test_accuracy
        self.test_loss = test_loss
        self.holdout_test_accuracy = holdout_test_accuracy
        self.holdout_test_loss = holdout_test_loss

        self.auc = auc
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.npv = npv
        self.ppv = ppv

        self.history = json.dumps(history, default=default)

        self.probabilities = json.dumps(probabilities, default=default)
        self.labels = json.dumps(labels, default=default)

        self.test_probabilities = json.dumps(test_probabilities, default=default)
        self.test_labels = json.dumps(test_labels, default=default)

        self.holdout_test_probabilities = json.dumps(holdout_test_probabilities, default=default)
        self.holdout_test_labels = json.dumps(holdout_test_labels, default=default)

        self.description = description
        self.input_form = input_form
        self.label_form = label_form

        self.hyperparameters = json.dumps(hyperparameters)

        self.test_f1_result = test_f1_result
        self.holdout_f1_result = holdout_f1_result

    def dict(self):
        return {
            "fold": self.fold,
            "total_folds": self.total_folds,
            "id": self.id,
            "split": self.split,
            "model": self.model,
            "createdOn": self.created_on.timestamp(),
            "trainDataStats": json.loads(self.train_data_stats),
            "validationDataStats": json.loads(self.validation_data_stats),
            "trainAccuracy": self.train_accuracy,
            "accuracy": self.accuracy,
            "input_form": self.input_form,
            "label_form": self.label_form,
        }

    def get_holdout_probabilities(self):
        return json.loads(self.holdout_test_probabilities)

    def get_holdout_labels(self):
        return json.loads(self.holdout_test_labels)

    def get_hyperparameters(self):
        return json.loads(self.hyperparameters)

class CalculatedResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_on = db.Column(db.DateTime, default=datetime.utcnow)
    split = db.Column(db.String)

    model = db.Column(db.String)

    min_loss_f1 = db.Column(db.Float)
    min_loss_accuracy = db.Column(db.Float)
    min_loss_test_acc = db.Column(db.Float)
    min_loss_test_loss = db.Column(db.Float)

    max_acc_f1 = db.Column(db.Float)
    max_acc_accuracy = db.Column(db.Float)
    max_acc_test_acc = db.Column(db.Float)
    max_acc_test_loss = db.Column(db.Float)

    averaged_test_f1 = db.Column(db.Float)
    averaged_test_accuracy = db.Column(db.Float)
    averaged_test_loss = db.Column(db.Float)

    averaged_holdout_f1 = db.Column(db.Float)
    averaged_holdout_accuracy = db.Column(db.Float)
    averaged_holdout_loss = db.Column(db.Float)

    hyperparameters = db.Column(db.String)
    description = db.Column(db.String)
    input_form = db.Column(db.String)

    def __init__(self,
            split,
            model,
            min_loss_f1,
            min_loss_accuracy,
            min_loss_test_acc,
            min_loss_test_loss,
            max_acc_f1,
            max_acc_accuracy,
            max_acc_test_acc,
            max_acc_test_loss,
            averaged_test_f1,
            averaged_test_accuracy,
            averaged_test_loss,
            averaged_holdout_f1,
            averaged_holdout_accuracy,
            averaged_holdout_loss,
            hyperparameters,
            description,
            input_form
            ):
        self.split = split
        self.model = model
        self.min_loss_f1 = min_loss_f1
        self.min_loss_accuracy = min_loss_accuracy
        self.min_loss_test_acc = min_loss_test_acc
        self.min_loss_test_loss = min_loss_test_loss
        self.max_acc_f1 = max_acc_f1
        self.max_acc_accuracy = max_acc_accuracy
        self.max_acc_test_acc = max_acc_test_acc
        self.max_acc_test_loss = max_acc_test_loss
        self.averaged_test_f1 = averaged_test_f1
        self.averaged_test_accuracy = averaged_test_accuracy
        self.averaged_test_loss = averaged_test_loss
        self.averaged_holdout_f1 = averaged_holdout_f1
        self.averaged_holdout_accuracy = averaged_holdout_accuracy
        self.averaged_holdout_loss = averaged_holdout_loss
        self.description = description
        self.input_form = input_form
        self.hyperparameters = json.dumps(hyperparameters)

    def dict(self):
        return {
            "id": self.id,
            "split": self.split,
            "model": self.model,
            "createdOn": self.created_on.timestamp(),
            "input_form": self.input_form,
            "hyperparameters": json.loads(self.hyperparameters),
            "description": self.description,
            "min_loss_f1": self.min_loss_f1,
            "min_loss_accuracy": self.min_loss_accuracy,
            "min_loss_test_acc": self.min_loss_test_acc,
            "min_loss_test_loss": self.min_loss_test_loss,
            "max_acc_f1": self.max_acc_f1,
            "max_acc_accuracy": self.max_acc_accuracy,
            "max_acc_test_acc": self.max_acc_test_acc,
            "max_acc_test_loss": self.max_acc_test_loss,
            "averaged_test_f1": self.averaged_test_f1,
            "averaged_test_accuracy": self.max_acc_accuracy,
            "averaged_test_loss": self.averaged_test_loss,
            "averaged_holdout_f1": self.averaged_holdout_f1,
            "averaged_holdout_accuracy": self.averaged_holdout_accuracy,
            "averaged_holdout_loss": self.averaged_holdout_loss,
        }

    def get_hyperparameters(self):
        return json.loads(self.hyperparameters)
