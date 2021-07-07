import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
import functools
# import wrapt
from typing import List, Iterator, Tuple

from ..core import Operator, DataDict


def make_val_and_grad_fn(value_fn):
    @functools.wraps(value_fn)
    def val_and_grad(x):
        return tfp.math.value_and_gradient(value_fn, x)
    return val_and_grad


# def val_and_grad():
#     @wrapt.decorator
#     def wrapper(wrapped, instance, args, kwargs):
#         return tfp.math.value_and_gradient(wrapped, *args, **kwargs)
#     return wrapper


class Dkl(Operator):
    def __init__(self, indx_list, weight_col='W', **kwargs):
        assert len(indx_list) != 0
        self.indx_list = indx_list
        self.weight_col = weight_col

        self.train_weights = None
        self.weight_params = None
        self.status = True

    @staticmethod
    def apply_to_member(data, weights_df):
        df_with_weights = pd.concat([data.data, weights_df], axis=1, sort=False)
        data_new = data.copy(data=df_with_weights)
        return data_new

    def y_forward(self, y: DataUnit, x: DataUnit = None, x_frwd: DataUnit = None) -> DataUnit:
        y_train_new = None
        if y['train'] is not None:
            y_train_new = self.apply_to_member(y['train'], self.train_weights)

        y_valid_new = None
        if y['valid'] is not None:
            val_weights = pd.DataFrame(data=np.ones(y['valid'].data.shape[0]) / y['valid'].data.shape[0],
                                       index=y['valid'].data.index,
                                       columns=[self.weight_col])
            y_valid_new = self.apply_to_member(y['valid'], val_weights)

        y_test_new = None
        if y['test'] is not None:
            val_weights = pd.DataFrame(data=np.ones(y['test'].data.shape[0]) / y['test'].data.shape[0],
                                       index=y['test'].data.index,
                                       columns=[self.weight_col])
            y_test_new = self.apply_to_member(y['test'], val_weights)

        return DataUnit(train=y_train_new, valid=y_valid_new, test=y_test_new)

    # @ApplyToDataUnit()
    # def y_backward(self, y_frwd: Data) -> Data:
    #     '''Probably it should remove Weight column'''
    #     return y_frwd

    def _fit_(self, x: DataUnit, y: DataUnit) -> None:
        if not isinstance(x, DataUnit):
            raise Exception(f'Error: Dkl only works with DataUnit, not with {x.__class__.__name__}')
        self.calc_weights(x['train'].data, x['test'].data)
        return None

    @staticmethod
    def scale_data(data, mu, sigma):
        scaled = (data - mu) / sigma 
        return scaled
    
    def prepare_data(self, train: pd.DataFrame, valid: pd.DataFrame):
        """train: n*k, valid:n1*k
            kernel_matrix: n*m
            moments: 1*m"""
        train = tf.convert_to_tensor(train.to_numpy(), dtype=tf.float64)
        valid = tf.convert_to_tensor(valid.to_numpy(), dtype=tf.float64)

        all_data = tf.concat([train, valid], axis=0)
        
        mu = tf.math.reduce_mean(all_data, axis=0, keepdims=True)
        sigma = tf.math.reduce_std(all_data, axis=0, keepdims=True)
        
        scaled_train = self.scale_data(train, mu, sigma)
        scaled_valid = self.scale_data(valid, mu, sigma)

        kernel_matrix = []
        moments = []
        for (idx, func) in self.indx_list:
            kernel_matrix.append(func(scaled_train[:, idx]))
            moments.append(tf.math.reduce_mean(func(scaled_valid[:, idx]), axis=0))
        
        kernel_matrix = tf.convert_to_tensor(kernel_matrix, dtype=tf.float64)
        kernel_matrix = tf.transpose(kernel_matrix)
        
        moments = tf.convert_to_tensor(moments, dtype=tf.float64)
        return kernel_matrix, moments
    
    @staticmethod
    def calculate_P(alpha, kernel_matrix):
        """matrix: n*m
        alpha: m*1
        P: n*1"""
        matrix_alpha = tf.linalg.tensordot(kernel_matrix, alpha, axes=1)
        P = tf.math.exp(matrix_alpha - tf.math.reduce_mean(matrix_alpha, keepdims=True))
        P /= tf.math.reduce_sum(P, keepdims=True)
        return P
    
    @staticmethod
    def data_eff(P):
        Hp = -tf.linalg.tensordot(P, tf.math.log(P), axes=1)
        data_eff = tf.math.exp(Hp) / P.shape[0]
        return data_eff.numpy()
    
    @staticmethod
    def Dkl(P):
        Hp = tf.linalg.tensordot(P, tf.math.log(P), axes=1)
        Huniform = tf.math.log(tf.constant(P.shape[0], tf.float64))
        dkl = Hp + Huniform
        return dkl.numpy()

    @staticmethod
    def check_P(P):
        status = False
        if tf.math.reduce_all(tf.math.is_finite(P)).numpy():
            if 0.9 <= tf.math.reduce_sum(P).numpy() <= 1.1:
                status = True
        return status
    
    @staticmethod
    def check_solution(res):
        status = False
        if res.converged:
            if res.objective_value.numpy() < 10e-3:
                if tf.norm(res.position).numpy() < 10e3:
                    status = True
        return status

    def calc_weights(self, train, valid):
        @make_val_and_grad_fn
        def opt_func(alpha):
            """matrix: n*m
            alpha: m*1
            moments: 1*m
            """
            matrix_alpha = tf.linalg.tensordot(kernel_matrix, alpha, axes=1)
            P = tf.math.exp(matrix_alpha - tf.math.reduce_max(matrix_alpha, keepdims=True))
            P /= tf.math.reduce_sum(P, keepdims=True)
            opt_mtrx = tf.linalg.tensordot(P, kernel_matrix, axes=1) - moments
            sq_opt_mtrx = tf.math.square(opt_mtrx) 
            return tf.math.reduce_sum(sq_opt_mtrx)
        
        kernel_matrix, moments = self.prepare_data(train, valid)
        init_point = tf.zeros([len(self.indx_list), ], dtype=tf.float64)
        optim_results = tfp.optimizer.bfgs_minimize(opt_func, init_point, parallel_iterations=4, max_iterations=100)

        """Check yourself before you..."""

        if not self.check_solution(optim_results):
            self.status = False
            P = tf.ones([train.shape[0],], dtype=tf.float64) / train.shape[0]
        else:
            P = self.calculate_P(optim_results.position, kernel_matrix)
            if not self.check_P(P):
                self.status = False
                P = tf.ones([train.shape[0],], dtype=tf.float64) / train.shape[0]

        self.train_weights = pd.DataFrame(data=P.numpy() * P.numpy().shape[0],
                                          index=train.index,
                                          columns=[self.weight_col])

        data_efficiency = self.data_eff(P)
        dkl = self.Dkl(P)
        
        self.weight_params = {'DataEfficiency': data_efficiency, 'Dkl': dkl}

        print('BFGS Results')
        print('Converged:', optim_results.converged)
        print('Obj value: ', optim_results.objective_value)
        print('Number of iterations:', optim_results.num_iterations)
        print('Location of the minimum:', optim_results.position)
