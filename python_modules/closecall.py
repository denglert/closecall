#!/usr/bin/env python

import sys
import os
import multiprocessing
import numpy as np
import json


DEBUG = False


class Model:

    def __init__(self, log_likelihood_function, data):
        
        self.log_likelihood_function = log_likelihood_function  
        self.data = data

        print()
        print('Model should be initialised.')

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
            #print("data: {}".format(data))
            print("ll {}".format(ll))
            print("Inside Model.log_likelihood()")

        return ll



class Sampler:

    def __init__(self, model, proposal_function, initial_theta, config):

        self.model = model
        self.proposal_function = proposal_function
        self.initial_theta = initial_theta

        self.header = ' '.join(['{}'.format(k) for k in self.initial_theta._fields]) + ' ' + 'log_likelihood\n'

        self.theta_chain = []
        self.log_likelihood_chain = []

        ll_initial = self.log_likelihood(self.initial_theta, self.model.data)

        if DEBUG:
            print(ll_initial)

        self.theta_chain.append(initial_theta)
        self.log_likelihood_chain.append(ll_initial)

        self.config = config

        self.config_setup()

        
        print()
        print('Sampler should be initialised.')


    @classmethod
    def from_config_file_path(cls):
        pass


    def config_setup(self):

        if self.config['write_to_output_file_stream'] and self.config['output_file_path']:
            self.output_file_stream = open(self.config['output_file_path'], 'w')
            self.write_output_file_header()

    def write_output_file_header(self):
        self.output_file_stream.write(self.header)

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


    def write_to_output_file_stream(self):
        theta = self.theta_chain[-1]
        ll    = self.log_likelihood_chain[-1]

        theta_str = ' '.join(['{}'.format(getattr(theta, k)) for k in theta._fields])
        line = "{} {}\n".format(theta_str, ll)

        self.output_file_stream.write(line)

    def write_to_stdout_stream(self):
        theta = self.theta_chain[-1]
        ll    = self.log_likelihood_chain[-1]
        line = "{} {}".format(theta, ll)
        print(line)


    def update(self):

        self.update_rule()

        if self.config['write_to_output_file_stream']:
            self.write_to_output_file_stream()

        if self.config['write_to_stdout_stream']:
            self.write_to_stdout_stream()


    
class MetropolisHastings(Sampler):


    def __init__(self, model, proposal_function, initial_theta, config):
        super().__init__(model, proposal_function, initial_theta, config)
        self.accepted = 0

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
        r = np.exp(ll_proposed-ll_previous)

        if u < r:
            self.model.set_theta(theta_proposed)
            self.theta_chain.append(theta_proposed)
            self.log_likelihood_chain.append(ll_proposed)
            self.accepted += 1
        else:
            self.model.set_theta(theta_previous)
            self.theta_chain.append(theta_previous)
            self.log_likelihood_chain.append(ll_previous)



class HMC(Sampler):

    def update_rule(self):
        pass



#class FileStream:
#
#    def __init__(self, output_file_path):
#        self.output_stream = open(output_file_path, 'w')
#
#    def write_line(self, line):
#        output_stream.writeline(line)
#
#    def close(self):
#        output_stream.close()
