#!/usr/bin/env python2

import argparse
import simpy
import pandas as pd
import numpy as np
import sys, os
import abc
import random
import json

# default cmdline args
cmd_parser = argparse.ArgumentParser()
cmd_parser.add_argument('--config', type=str, help='JSON config file to control the simulations', required=True)

class Logger(object):
    debug = True
    def __init__(self, env):
        self.env = env

    @staticmethod
    def init_params():
        pass

    def log(self, s):
        if Logger.debug:
            print '{}: {}'.format(self.env.now, s)


def DistGenerator(dist, **kwargs):
    if dist == 'bimodal':
        bimodal_samples = map(int, list(np.random.normal(kwargs['lower_mean'], kwargs['lower_stddev'], kwargs['lower_samples']))
                                   + list(np.random.normal(kwargs['upper_mean'], kwargs['upper_stddev'], kwargs['upper_samples'])))
    while True:
        if dist == 'uniform':
            yield random.randint(kwargs['min'], kwargs['max'])
        elif dist == 'normal':
            yield int(np.random.normal(kwargs['mean'], kwargs['stddev']))
        elif dist == 'poisson':
            yield np.random.poisson(kwargs['lambda']) 
        elif dist == 'lognormal':
            yield int(np.random.lognormal(kwargs['mean'], kwargs['sigma']))
        elif dist == 'exponential':
            yield kwargs['offset'] + int(np.random.exponential(kwargs['lambda']))
        elif dist == 'fixed':
            yield kwargs['value']
        elif dist == 'bimodal':
            yield random.choice(bimodal_samples)
        else:
            print 'ERROR: Unsupported distrbution: {}'.format(dist)
            sys.exit(1)

class NanoSimulator(object):
    """This class controls the simulation"""
    config = {} # user specified input
    out_dir = 'out'
    out_run_dir = 'out/run-0'
    def __init__(self, env):
        self.env = env
        self.logger = Logger(self.env)

        self.local_task_time = NanoSimulator.config['local_task_time'].next()
        self.num_tasks = NanoSimulator.config['num_tasks'].next()
        # initialize RPC completion time distribution
        self.rpc_time = NanoSimulator.config['rpc_time'].next()
        kwargs = {}
        if self.rpc_time == 'uniform':
            kwargs['min'] = NanoSimulator.config['rpc_time_min'].next()
            kwargs['max'] = NanoSimulator.config['rpc_time_max'].next()
        elif self.rpc_time == 'normal':
            kwargs['mean'] = NanoSimulator.config['rpc_time_mean'].next()
            kwargs['stddev'] = NanoSimulator.config['rpc_time_stddev'].next()
        elif self.rpc_time == 'poisson':
            kwargs['lambda'] = NanoSimulator.config['rpc_time_lambda'].next()
        elif self.rpc_time == 'lognormal':
            kwargs['mean'] = NanoSimulator.config['rpc_time_mean'].next()
            kwargs['sigma'] = NanoSimulator.config['rpc_time_sigma'].next()
        elif self.rpc_time == 'exponential':
            kwargs['offset'] = NanoSimulator.config['rpc_time_offset'].next()
            kwargs['lambda'] = NanoSimulator.config['rpc_time_lambda'].next()
        elif self.rpc_time == 'fixed':
            kwargs['value'] = NanoSimulator.config['rpc_time_value'].next()
        elif self.rpc_time == 'bimodal':
            kwargs['lower_mean'] = NanoSimulator.config['rpc_time_lower_mean'].next()
            kwargs['lower_stddev'] = NanoSimulator.config['rpc_time_lower_stddev'].next()
            kwargs['lower_samples'] = NanoSimulator.config['rpc_time_lower_samples'].next()
            kwargs['upper_mean'] = NanoSimulator.config['rpc_time_upper_mean'].next()
            kwargs['upper_stddev'] = NanoSimulator.config['rpc_time_upper_stddev'].next()
            kwargs['upper_samples'] = NanoSimulator.config['rpc_time_upper_samples'].next()
        self.rpc_time_dist = DistGenerator(self.rpc_time, **kwargs)

        # initialize run logs
        self.num_cores = []
        self.completion_time = []
        
        self.env.process(self.run_sim())

    def log(self, s):
        self.logger.log(s)

    def run_sim(self):
        # One run of the simulation will sweep all possible number of cores and record total completion time
        for c in range(1, self.num_tasks+1):
            self.num_cores.append(c)
            ct = yield self.env.process(self.get_completion_time(c))
            self.completion_time.append(ct)

    def get_completion_time(self, num_cores):
        # Do one iteration of the simulation and utilize the specified number of cores.
        # That is, do num_cores-1 RPCs and perform the rest of the tasks locally.
        num_remote_cores = num_cores - 1
        start_time = self.env.now
        events = []
        # RPC tasks
        for i in range(num_remote_cores):
            events.append(self.env.timeout(self.rpc_time_dist.next()))
        # Local tasks
        local_proc_time = (self.num_tasks - num_remote_cores)*self.local_task_time
        events.append(self.env.timeout(local_proc_time))
        # wait for all tasks to complete
        self.log('Starting to wait for all tasks ...')
        yield simpy.events.AllOf(self.env, events)
        ct = self.env.now - start_time
        self.log('All tasks complete! Completion Time = {} ns'.format(ct))
        self.env.exit(ct)

    def dump_run_logs(self):
        """Dump any logs recorded during this run of the simulation"""
        out_dir = os.path.join(os.getcwd(), NanoSimulator.out_run_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # log the measured request completion times
        df = pd.DataFrame({'num_cores':self.num_cores, 'completion_time':self.completion_time})
        write_csv(df, os.path.join(NanoSimulator.out_run_dir, 'completion_times.csv'))

def write_csv(df, filename):
    with open(filename, 'w') as f:
            f.write(df.to_csv(index=False))

def param(x):
    while True:
        yield x

def param_list(L):
    for x in L:
        yield x

def parse_config(config_file):
    """ Convert each parameter in the JSON config file into a generator
    """
    with open(config_file) as f:
        config = json.load(f)

    for p, val in config.iteritems():
        if type(val) == list:
            config[p] = param_list(val)
        else:
            config[p] = param(val)

    return config

def run_nano_sim(cmdline_args, *args):
    NanoSimulator.config = parse_config(cmdline_args.config)
    # make sure output directory exists
    NanoSimulator.out_dir = NanoSimulator.config['out_dir'].next()
    out_dir = os.path.join(os.getcwd(), NanoSimulator.out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # copy config file into output directory
    os.system('cp {} {}'.format(cmdline_args.config, out_dir))
    # run the simulations
    run_cnt = 0
    try:
        while True:
            print 'Running simulation {} ...'.format(run_cnt)
            # initialize random seed
            random.seed(1)
            np.random.seed(1)
            NanoSimulator.out_run_dir = os.path.join(NanoSimulator.out_dir, 'run-{}'.format(run_cnt))
            run_cnt += 1
            env = simpy.Environment()
            s = NanoSimulator(env, *args)
            env.run()
            s.dump_run_logs()
    except StopIteration:
        print 'All Simulations Complete!'

def main():
    args = cmd_parser.parse_args()
    # Run the simulation
    run_nano_sim(args)

if __name__ == '__main__':
    main()

