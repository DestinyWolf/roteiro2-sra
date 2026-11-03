function obs_linha = expansao(obs, raio)
    [num_obs, _] = size(obs);

    obs_linha = zeros(num_obs, 7);
    for i=1:num_obs
        switch(obs(i,1))
            case 0
                pos = obs(i, 2:3);
                l = obs(i, 4) + (2*raio);
                h = obs(i, 5) + (2*raio);
                obs_linha(i,1:5) = [0, pos(1), pos(2), l, h];
            case 1
                obs_linha(i) = obs(i); %ainda não sei o que vou fazer com o calc_alt_triangulo
            case 2
                r = obs(i, 2) * raio;
                pos = obs(i, 3:4);
                obs_linha(i, 1:4) = [2, r, pos(1), pos(2)];
        end
    end
end

function h = calc_alt_triangulo(obs)
    a = [obs(2), obs(5)];
    b = [obs(3), obs(6)];
    c = [obs(4), obs(7)];

    base_tri = b-a;
    comp_base = norm(base_tri);

    vetor_ab = [b - a, 0];
    vetor_ac = [c - a, 0];

    area =0.5*norm(cross(vetor_ab, vetor_ac));

    h = (2*area)/comp_base;
end
