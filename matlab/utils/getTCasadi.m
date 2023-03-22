function T=getTCasadi(t,deg)
  %For whatever reason, ([t.^[obj.p:-1:1] 1])' does not work properly with
  %other functions
    T=[];
    for i=deg:-1:0
        T=[T;t^i];
    end
end