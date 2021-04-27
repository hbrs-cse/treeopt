inputName = 'input.txt';
outputName = 'output.txt';

fileID = fopen(inputName, 'r');
X = textscan(fileID, '%f');
X = X{1};
fclose(fileID);

Y = sum(X.^2);

fileID = fopen('output.txt','w');
fprintf(fileID, '%f', Y);
fclose(fileID);