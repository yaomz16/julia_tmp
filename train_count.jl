using DifferentialEquations
using Flux
using DiffEqFlux
using Plots
using NPZ
using Zygote
using BSON: @save
using BSON: @load



################################################################################
### Hyper-parameters ###
input_start_num = 30 # start of input number, minimum is 0
input_end_num = 40 # end of input number
max_data_num = 80 # how many steps are there in total? max_data_num-input_start_num+1
numpts = 161
figure_output_freq = 2000



################################################################################
### Getting true solutions ###
using PyCall
w = zeros(numpts,numpts,max_data_num-input_start_num+1) # all the true solution
w_train = zeros(numpts,numpts,input_end_num-input_start_num+1) # the input data for training
t = zeros(max_data_num-input_start_num+1) # time matrix
t_train = zeros(input_end_num-input_start_num+1) # the input timesteps for training
py"""
import pandas as pd
import numpy as np
import julia
julia.install()
from julia import Main
numpts = Main.numpts
def get_w(x):
    name = 'snow_1_'+str(x)+'.csv'
    data=pd.read_csv(name)
    data=data.sort_values(by=['Points:0','Points:1'])
    a=np.array(data['Points:0'])
    a=a.reshape(numpts,numpts)
    b=np.array(data['Points:1'])
    b=b.reshape(numpts,numpts)
    w=np.array(data['w'])
    w=w.reshape(numpts,numpts)
    return w
def get_t(x):
    name = 'snow_1_'+str(x)+'.csv'
    data=pd.read_csv(name)
    data=data.sort_values(by=['Points:0','Points:1'])
    t = data['Time'][0]
    return t
with open('counter.txt','w') as fin:
    fin.write('')
"""

for i = input_start_num:max_data_num
    w_now = py"get_w"(i)
    w_now = reshape(w_now,numpts, numpts, 1)
    w[:,:,i-input_start_num+1] = w_now
    t_now = py"get_t"(i)
    t[i-input_start_num+1] = t_now
end

for i = input_start_num:input_end_num
    w_now = py"get_w"(i)
    w_now = reshape(w_now,numpts, numpts, 1)
    w_train[:,:,i-input_start_num+1] = w_now
    t_now = py"get_t"(i)
    t_train[i-input_start_num+1] = t_now
end

w0 = reshape(w[:,:,1],numpts, numpts, 1, 1) # Initial Conditions

tspan_train = (t_train[1],t_train[input_end_num-input_start_num+1])
print("\n\n\n")
print(tspan_train)
print("\n\n\n\n\n\n\n")



################################################################################
### Define architecture ###
dudt = Chain(Conv((5,5), 1=>4, swish, pad=2),
             Conv((1,1), 4=>1, swish, pad=0))



################################################################################
### NODE Setup ###
# n_ode = NeuralODE(dudt,tspan_train,Tsit5(),saveat=t_train,reltol=1e-7,abstol=1e-9)
@load "n_ode.bson" n_ode

function predict_n_ode()
  n_ode(w0)
end

loss_n_ode() = sum(abs2, w_train .- reshape(Array(predict_n_ode()),numpts,numpts,input_end_num-input_start_num+1))

### Training ###
data = Iterators.repeated((), figure_output_freq)
opt = ADAM(0.01)

counter = 0

cb = function ()
  display(loss_n_ode())
  print("\n")
  @save "n_ode.bson" n_ode
  global counter
  counter = counter + 1
  io = open("counter.txt", "a")
  now_loss = loss_n_ode()
  write(io, string(counter,"  ",now_loss))
  write(io,"\n")
  close(io)
end

ps = Flux.params(n_ode)
Flux.train!(loss_n_ode, ps, data, opt, cb = cb)
print("\n\n")
@save "n_ode.bson" n_ode
weights = params(n_ode)
@save "mymodel.bson" weights



################################################################################
### Prediction ###
@load "mymodel.bson" weights
tspan_all = (t[1],t[max_data_num-input_start_num+1])
n_ode_new = NeuralODE(dudt,tspan_all,Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9)
Flux.loadparams!(n_ode_new,weights)
predicted_results = Array(n_ode_new(w0))

# output predicted results
using Printf
for i = input_start_num:max_data_num
    pred_now = reshape(predicted_results[:,:,1,1,i-input_start_num+1], numpts, numpts)
    name_now = string("snow_predicted_",i,".txt")
    io = open(name_now,"w")
    for j = 1:numpts
        for k = 1:numpts
            @printf(io,"%.5f ", pred_now[j,k])
        end
        @printf(io,"\n")
    end
end



################################################################################
### ALL SET! ###
print("Exec End. Exit Now.\n")
