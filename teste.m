% =========================================================================
% PLANEJAMENTO DE ROTA POR CAMPOS DE POTENCIAL ARTIFICIAL (APF)
% =========================================================================
clear; clc; close all;

% 1. Configurações do Espaço e Posições
q_start =[0 0];       % Posição Inicial [x, y]
q_goal  = [10 10];     % Destino [x, y]

% Obstáculos: [x, y, raio, raio_de_influencia_rho0]
obstacles = [
    5, 5, 1.0, 2.0;
    7, 4, 0.8, 1.5
];

% 2. Ganhos e Parâmetros de Controle
xi = 0.5;         % Ganho atrativo
eta = 15;         % Ganho repulsivo
step_size = 0.1;  % Tamanho do passo (velocidade de avanço)
max_iter = 500;   % Limite de iterações para evitar loops infinitos
tolerance = 0.2;  % Distância mínima para considerar que chegou ao destino

% 3. Loop de Simulação da Rota
q = q_start;
path = q;
iter = 1;

while (norm(q - q_goal) > tolerance) && (iter < max_iter)
    
    % --- FORÇA ATRATIVA ---
    % Gradiente do potencial atrativo: xi * (q - q_goal)
    F_att = -xi * (q - q_goal);
    
    % --- FORÇA REPULSIVA ---
    F_rep = -1;
    for i = 1:size(obstacles, 1)
        obs_pos = obstacles(i, 1:2);
        obs_radius = obstacles(i, 3);
        rho0 = obstacles(i, 4);
        
        % Distância do robô até a superfície do obstáculo
        dist_to_center = norm(q - obs_pos);
        dist_to_obs = dist_to_center - obs_radius;
        
        if dist_to_obs <= 0
            % Colisão direta (força infinita para empurrar para fora)
            dist_to_obs = 0.01; 
        end
        
        % Aplica repulsão apenas se estiver dentro do raio de influência
        if dist_to_obs < rho0
            % Vetor direção do obstáculo para o robô
            dir_vector = (q - obs_pos) / dist_to_center;
            
            % Magnitude da força de repulsão (Gradiente de U_rep)
            rep_mag = eta * (1/dist_to_obs - 1/rho0) * (1 / (dist_to_obs^2));
            F_rep = F_rep + rep_mag * dir_vector;
        end
    end
    
    % --- FORÇA RESULTANTE E ATUALIZAÇÃO ---
    F_total = F_att + F_rep;
    
    % Normaliza a força para garantir passos de tamanho constante
    if norm(F_total) > 0
        q = q + step_size * (F_total / norm(F_total));
    end
    
    % Salva a rota
    path = [path; q];
    iter = iter + 1;
end

% 4. Visualização do Resultado
figure('Color', 'w');
hold on; grid on; axis equal;
xlim([0 12]); ylim([0 12]);
xlabel('Eixo X'); ylabel('Eixo Y');
title('Planejamento de Rota - Campos de Potencial');

% Desenha o Início e o Objetivo
plot(q_start(1), q_start(2), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
text(q_start(1)+0.2, q_start(2), 'Início');
plot(q_goal(1), q_goal(2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
text(q_goal(1)+0.2, q_goal(2), 'Destino');

% Desenha os Obstáculos e Áreas de Influência
theta = linspace(0, 2*pi, 100);
for i = 1:size(obstacles, 1)
    % Obstáculo físico (Sólido)
    fill(obstacles(i,1) + obstacles(i,3)*cos(theta), ...
         obstacles(i,2) + obstacles(i,3)*sin(theta), 'k', 'FaceAlpha', 0.6);
    % Zona de influência (Pontilhado)
    plot(obstacles(i,1) + obstacles(i,4)*cos(theta), ...
         obstacles(i,2) + obstacles(i,4)*sin(theta), 'r--');
end

% Desenha a Rota Gerada
plot(path(:,1), path(:,2), 'b-', 'LineWidth', 2);
legend('Início', 'Destino', 'Obstáculo', 'Zona de Influência', 'Rota Calculada', 'Location', 'best');