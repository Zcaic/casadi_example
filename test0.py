from casadi import *

class Example4To3(Callback):
  def __init__(self, name, opts={}):
    Callback.__init__(self)
    self.construct(name, opts)

  def get_n_in(self): return 1
  def get_n_out(self): return 1

  def get_sparsity_in(self,i):
    return Sparsity.dense(4,1)

  def get_sparsity_out(self,i):
    return Sparsity.dense(3,1)

  # Evaluate numerically
  def eval(self, arg):
    a,b,c,d = vertsplit(arg[0])
    ret = vertcat(sin(c)*d+d**2,2*a+c,b**2+5*c)
    return [ret]


class Example4To3_Fwd(Example4To3):
  def has_forward(self,nfwd):
    # This example is written to work with a single forward seed vector
    # For efficiency, you may allow more seeds at once
    return nfwd==1
  def get_forward(self,nfwd,name,inames,onames,opts):
    
    class ForwardFun(Callback):
      def __init__(self, opts={}):
        Callback.__init__(self)
        self.construct(name, opts)

      def get_n_in(self): return 3
      def get_n_out(self): return 1

      def get_sparsity_in(self,i):
        if i==0: # nominal input
          return Sparsity.dense(4,1)
        elif i==1: # nominal output
          return Sparsity(3,1)
        else: # Forward seed
          return Sparsity.dense(4,1)

      def get_sparsity_out(self,i):
        # Forward sensitivity
        return Sparsity.dense(3,1)

      # Evaluate numerically
      def eval(self, arg):
        a,b,c,d = vertsplit(arg[0])
        a_dot,b_dot,c_dot,d_dot = vertsplit(arg[2])
        print("Forward sweep with", a_dot,b_dot,c_dot,d_dot)
        w0 = sin(c)
        w0_dot = cos(c)*c_dot
        w1 = w0*d
        w1_dot = w0_dot*d+w0*d_dot
        w2 = d**2
        w2_dot = 2*d_dot*d
        r0 = w1+w2
        r0_dot = w1_dot + w2_dot
        w3 = 2*a
        w3_dot = 2*a_dot
        r1 = w3+c
        r1_dot = w3_dot+c_dot
        w4 = b**2
        w4_dot = 2*b_dot*b
        w5 = 5*w0
        w5_dot = 5*w0_dot
        r2 = w4+w5
        r2_dot = w4_dot + w5_dot
        ret = vertcat(r0_dot,r1_dot,r2_dot)
        return [ret]
    # You are required to keep a reference alive to the returned Callback object
    self.fwd_callback = ForwardFun()
    return self.fwd_callback

    
f = Example4To3_Fwd('f')
x = MX.sym("x",4)
J = Function('J',[x],[jacobian(f(x),x)])
print(J(vertcat(1,2,0,3)))