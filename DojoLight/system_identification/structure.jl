################################################################################
# structure
################################################################################
# measurement obtained at each time step
abstract type Measurement{T} end

# time-varying state of the system
abstract type State{T} end

# parameters of the system that are fixed in time and that we want to estimate
abstract type Parameters{T} end

# all ground-truth information including time-invariant parameters that we want to estimate (parameters)
# and that we assume are known (camera position for instance).
# it also contains the state of the system
abstract type Context{T} end

# objective that we want to minimize to regress correct states and parameters
abstract type Objective{T} end

# find the best fitting states and parameters given a sequence of measurement and a prior over the state and parameters
# it contains the context and the objective
abstract type Optimizer{T} end
