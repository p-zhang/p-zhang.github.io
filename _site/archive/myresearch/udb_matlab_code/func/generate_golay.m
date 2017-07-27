function [a,b] = generate_golay(N)
% [a,b] = generate_golay(N)
%
% Generate the Golay codes a and b with length 2^N.
%
% Then write them to disk as golayA.wav and golayB.wav.


% These initial a and b values are Golay
a = [1 1];
b = [1 -1];

% Iterate to create a longer Golay sequence
while (N>1)
    olda = a;
    oldb = b;
    a = [olda oldb];
    b = [olda -oldb];

    N = N - 1;
end
