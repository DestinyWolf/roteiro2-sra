
%% Função criada para exibir o mapa
function plotar_mapa(obstaculos, robx, roby)
    [num_obs, _] = size(obstaculos);

    figure(1);
    hold on;
    for i=1:num_obs
        switch(obstaculos(i,1))
            case 0 %quadrado
                pos = obstaculos(i, 2:3);
                l = obstaculos(i, 4);
                h = obstaculos(i, 5);
                rectangle('Position', [pos(1)-l/2, pos(2)-h/2, l, h], 'EdgeColor', 'b', 'LineWidth', 2);
            case 1 % triangulo
                x = obstaculos(i, 2:4);
                y = obstaculos(i, 5:7);
                x(end+1) = x(1);
                y(end+1) = y(1);
                plot(x, y, 'r-', 'LineWidth', 2);
            case 2
                raio = obstaculos(i, 2)/2;
                pos = obstaculos(i, 3:4);
                rectangle('Position', [raio, raio, pos(1)+(raio/2), pos(2)+(raio/2)], 'Curvature', [1 1], 'EdgeColor', 'g', 'LineWidth', 2);
        end
    end
    plot(robx, roby, 'k*', 'LineWidth', 2);
    axis('square')
    hold off;
    grid on;
end
