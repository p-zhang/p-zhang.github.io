function op = cconv(x,y)
%CCONV	N-point circular convolution
%
%	op = CCONV(x,y) performs the N-point circular convolution
%	of vectors x and y.  'op' is returned as a row vector.  x and y
% 	must be vectors of equal lengths.  
%
%	See also CONV

%
% Peter Mash
% pete@lightblueoptics.com
% 17-Nov-2006

Nx = length(x);
Ny = length(y);


% catch error - different lengths
if(Nx~=Ny)
    error('Vectors must be the same length in this version');
end

% catch error - x or y is a matrix not a vector
[N M] = size(x);
if((N<N*M) && (M<N*M))
    error('x must be a vector')
end
[N M] = size(y);
if((N<N*M) && (M<N*M))
    error('y must be a vector')
end

% perform cyclic convolution
for n=1:Nx
    op(n) = sum(x(1:Nx).*y(mod(n-1:-1:n-Ny,Ny)+1));
end

% output as a column vector
op = op.';

% note, the same result may be more quickly achieved using:
%
% op = ifft(fft(x).*fft(y));