function vct = tras264(input)

T = dlmread('Zig-Zag Pattern.txt');
T = T +1;
vct = zeros(1, 64);
for i = 1:8
    for j = 1:8
        vct(T(i, j)) = input(i, j);
    end
end

end