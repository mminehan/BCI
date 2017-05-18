function [featVec, ni1] = MovingWinFeats(x, fs, winLen, winDisp, featFn)

% % time in seconds
% t = 0.01:0.01:3;
% % Desired frequency of sine wave in Hz
% f = 2;
% % signal
% toy_signal = sin(2*pi*f*t);
% 
% x = toy_signal;
% fs = 100;
% winLen = 0.5;
% winDisp = 0.1;
% featFn = LLFn;

% Window displacement in number of points
disp = winDisp*fs;
len = winLen*fs;

% Number of windows
wins = floor(numel(x) - len + disp)/(disp);

ni1 = rem(numel(x) - len + disp, disp);
featVec = zeros(1,wins);
tempSig = x(ni1 +1:len);

for ii = 1:wins-1
    
    featVec(ii) = featFn(tempSig);
    tempSig = zeros(1,len);
    tempSig = x(ni1+1 + disp*ii:ni1 + len + disp*ii);
    
end
end