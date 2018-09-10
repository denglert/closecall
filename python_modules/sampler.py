#!/usr/bin/env python

import sys
import os
import multiprocessing
import numpy as np
import json


DEBUG = False

class Parameters:

    def __init__(self, theta):

        self.theta = theta

    def __getitem__(self, key):
        return self.theta[key]

    def __getattr__(self, name):
        return self.theta[name]


class Variables:

    def __init__(self, variables):
        self.variables = variables



class Model:

    def __init__(self, log_likelihood_function, data):
        
        self.log_likelihood_function = log_likelihood_function  
        self.data = data

    def set_theta(self, theta):
        self.theta = theta

    def set_theta_names(self, names):
        self.theta_names = names

    def set_vars(self, vars):
        self.vars = vars

    def log_likelihood(self, theta, data, vars):


        ll = self.log_likelihood_function(theta, data, vars)

        if DEBUG:
            print("Inside Model.log_likelihood()")
            print("theta: {}".format(theta))
            print("data: {}".format(data))
            print("ll {}".format(ll))
            print("Inside Model.log_likelihood()")

        return ll




class Sampler:

    def __init__(self, model, proposal_function, initial_theta, config):

        self.model = model
        self.proposal_function = proposal_function
        self.initial_theta = initial_theta

        self.theta_chain = []
        self.log_likelihood_chain = []

        ll_initial = self.log_likelihood(self.initial_theta, self.model.data)

        if DEBUG:
            print(ll_initial)

        self.theta_chain.append(initial_theta)
        self.log_likelihood_chain.append(ll_initial)

        self.config = config

    @classmethod
    def from_config_file_path(cls):
        pass

    def set_initial_theta(self, initial_theta):
        self.initial_theta = initial_theta


    def log_likelihood(self, theta, data):

        ll = self.model.log_likelihood(theta, data, self.model.vars)
        return ll


    def set_config(self, config):
        self.config = config

    def load_config_file(self, config_file_path):

        self.config_file_path = config_file_path
        with open(config_file_path, 'r') as config:
            self.config = json.load(config)


    def write_to_output_stream(self):
        theta = self.theta_chain[-1]
        ll    = self.log_likelihood_chain[-1]
        line = "{} {}".format(theta, ll)

        self.output_stream.writeline(line)

    def update(self):

        self.update_rule(self)

        if self.config['write_to_stream']:
            self.write_to_output_stream(stream)



    
class MetropolisHastings(Sampler):

    def update_rule(self):

        theta_previous = self.theta_chain[-1]
        theta_proposed = self.proposal_function(self.model.theta)

        ll_previous = self.log_likelihood_chain[-1]
        ll_proposed = self.log_likelihood(theta_proposed, self.model.data)

        if DEBUG:
            print("")
            print("Inside update()")
            print(ll_previous)
            print(ll_proposed)
            print("Inside update()")
            print("")
    

        u = np.random.uniform()
        r = ll_proposed/ll_proposed

        if u < r:
            self.model.set_theta(theta_proposed)
            self.theta_chain.append(theta_proposed)
            self.log_likelihood_chain.append(ll_proposed)
        else:
            self.model.set_theta(theta_previous)
            self.theta_chain.append(theta_previous)
            self.log_likelihood_chain.append(ll_previous)



class HMC(sampler):

    def update_rule(self):
        pass



class FileStream:

    def __init__(self, output_file_path):
        self.output_stream = open(output_file_path, 'w')

    def write_line(self, line):
        output_stream.writeline(line)

    def close(self):
        output_stream.close()
